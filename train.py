import torch
import cv2
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
from collections import OrderedDict
import json
from functools import reduce
from thop import profile
import copy

from model.model import T2VQA
from dataset.dataset import T2VDataset

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    # 构建视频序列
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, label = line_split
            label = float(label) # 得分
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))], # 训练集
        video_infos[int(ratio * len(video_infos)) :], # 测试集
    )

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

def finetune_epoch(
    ft_loader,
    model,
    optimizer,
    scheduler,
    device,
    epoch=-1,
):
    model.train()
    total_loss = 0.0  
    
    # Qwen 专用引导式填空 Prompt
    prompt_template = 'Carefully watch the video and evaluate its quality from the aspects of temporal consistency, aesthetic beauty, and semantic alignment. The overall quality of this video is'
    
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        
        # 组装传入 model 的数据字典，确保适配重构后 model.py 要求的设备存放
        inputs = {}
        inputs["video"] = data["video"].to(device)
        if "video_aesthetic" in data:
            inputs["video_aesthetic"] = data["video_aesthetic"].to(device)

        y = data["gt_label"].float().detach().to(device)
        caption = data['prompt']

        scores = model(inputs, caption=caption, prompt=prompt_template)

        y_pred = scores

        p_loss = plcc_loss(y_pred, y)
        r_loss = rank_loss(y_pred, y)

        loss = p_loss + 0.3 * r_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    model.eval()
    avg_loss = total_loss / len(ft_loader)  
    return avg_loss  

def inference_set(
    inf_loader,
    model,
    device,
    best_,
    save_model=False,
    suffix="s",
    save_name="divide",
    save_type="head",
):
    results = []
    best_s, best_p, best_k, best_r = best_

    prompt_template = 'Carefully watch the video and evaluate its quality from the aspects of temporal consistency, aesthetic beauty, and semantic alignment. The overall quality of this video is'

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        inputs, video_up = {}, {}

        inputs['video'] = data['video'].to(device)
        if "video_aesthetic" in data:
            inputs["video_aesthetic"] = data["video_aesthetic"].to(device)
            
        with torch.no_grad():
            caption = data['prompt']
            result["pr_labels"] = model(inputs, caption=caption, prompt=prompt_template).cpu().numpy()

            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del inputs, video_up
        results.append(result)

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    del results, result  
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                if (
                    "qformer" in key
                    or "motion_proj" in key
                    or "aesthetic_proj" in key
                    or "slowfast" in key
                    or "aesthetic_conv3d" in key
                    or "lora" in key # 【新增】：保存模型时带上 LoRA 权重
                ):
                    head_state_dict[key] = v
            print("Following keys are saved (for head-only):", head_state_dict.keys())
            torch.save(
                {"state_dict": head_state_dict, "validation_results": best_,},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )
        else:
            torch.save(
                {"state_dict": state_dict, "validation_results": best_,},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )

    best_s, best_p, best_k, best_r = (
        max(best_s, s),
        max(best_p, p),
        max(best_k, k),
        min(best_r, r),
    )

    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return (best_s, best_p, best_k, best_r), (s, p, k, r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="t2vqa.yml", help="the option file"
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="t2v", help="target_set"
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bests_ = []
    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1

    print(opt["split_seed"])

    for split in range(num_splits):
        model = T2VQA(opt["model"]["args"]).to(device)
        
        if opt.get("split_seed", -1) > 0:
            opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
            opt["data"]["eval"] = copy.deepcopy(opt["data"][args.target_set])

            split_duo = train_test_split(
                opt["data"][args.target_set]["args"]["data_prefix"],
                opt["data"][args.target_set]["args"]["anno_file"],
                seed=opt["split_seed"] * (split + 1),
            )
            (
                opt["data"]["train"]["args"]["anno_file"],
                opt["data"]["eval"]["args"]["anno_file"],
            ) = split_duo

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = T2VDataset(
                    opt["data"][key]["args"]
                )
                train_datasets[key] = train_dataset
                print(len(train_dataset.video_infos))

        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=opt["batch_size"],
                num_workers=opt["num_workers"],
                shuffle=True,
            )

        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("eval"):
                val_dataset = T2VDataset(
                    opt["data"][key]["args"]
                )
                print(len(val_dataset.video_infos))
                val_datasets[key] = val_dataset

        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=opt["num_workers"],
                pin_memory=True,
            )

        # 修改：精确解耦参数组，针对三分支架构 + LoRA
        param_groups = []   
        for name, param in model.named_parameters():
            if (
                "qformer" in name
                or "motion_proj" in name
                or "aesthetic_proj" in name
                or "slowfast" in name
                or "aesthetic_conv3d" in name
                or "lora" in name # 【新增】：确保 lora 权重参与梯度更新
            ):
                param.requires_grad = True
                param_groups += [
                        {"params": param, "lr": opt["optimizer"]["lr"]}
                ]
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )

        warmup_iter = 0 
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"]) * len(train_loader))

        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)

        bests = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
        history_log = []  
        
        # ================= 【新增】：断点续训 (Auto-Resume) 加载逻辑 =================
        start_epoch = 0
        os.makedirs("pretrained_weights", exist_ok=True)
        latest_ckpt_path = f"pretrained_weights/{opt['name']}_latest_{args.target_set}_{split}.pth"
        
        if os.path.exists(latest_ckpt_path):
            print(f"\n[*] 发现中断的训练状态: {latest_ckpt_path}。正在恢复...")
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            
            # 1. 恢复模型权重 (使用 strict=False，因为我们为了省空间只存了参与训练的权重)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            
            # 2. 恢复优化器和学习率调度器状态 (非常重要，否则动量和学习率会重置，导致Loss爆炸)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            
            # 3. 恢复进度和最佳指标
            start_epoch = checkpoint["epoch"] + 1
            bests = checkpoint.get("bests", bests)
            
            # 4. 恢复历史 JSON 日志避免被覆盖
            log_file = f"training_history_split{split}.json"
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    try:
                        history_log = json.load(f)
                    except Exception:
                        pass
                        
            print(f"[*] 成功从第 {checkpoint['epoch']} 个 epoch 恢复。下一个执行的 epoch 将是 {start_epoch}\n")
        # =========================================================================
        
        # 【修改】：将起始 epoch 从 0 改为 start_epoch
        for epoch in range(start_epoch, opt["num_epochs"]):
            print(f"End-to-end Epoch {epoch}:")
            epoch_train_loss = 0.0
            print(f"End-to-end Epoch {epoch}:")
            epoch_train_loss = 0.0  
            
            for key, train_loader in train_loaders.items():
                epoch_train_loss = finetune_epoch(
                    train_loader,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                )
            
            epoch_data = {
                "epoch": epoch,
                "train_loss": epoch_train_loss
            }

            for key in val_loaders:
                bests[key], current_metrics = inference_set(
                    val_loaders[key],
                    model,
                    device,
                    bests[key],
                    save_model=opt["save_model"],
                    save_name=opt["name"] + "_head_" + args.target_set + f"_{split}",
                    suffix=key + "_s",
                    save_type="head",
                )
                
                s, p, k, r = current_metrics
                epoch_data[f"val_{key}_SRCC"] = float(s)
                epoch_data[f"val_{key}_PLCC"] = float(p)
                epoch_data[f"val_{key}_KRCC"] = float(k)
                epoch_data[f"val_{key}_RMSE"] = float(r)
            
            history_log.append(epoch_data)
            
            with open(f"training_history_split{split}.json", "w", encoding="utf-8") as f:
                json.dump(history_log, f, indent=4)
            
           # ================= 【增强】：带有防爆盘机制的 Checkpoint 保存逻辑 =================
            head_state_dict = OrderedDict()
            for k_name, v_param in model.state_dict().items():
                if (
                    "qformer" in k_name
                    or "motion_proj" in k_name
                    or "aesthetic_proj" in k_name
                    or "slowfast" in k_name
                    or "aesthetic_conv3d" in k_name
                    or "lora" in k_name
                ):
                    head_state_dict[k_name] = v_param
                    
            latest_state = {
                "epoch": epoch,
                "state_dict": head_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "bests": bests
            }
            
            # 使用临时文件进行原子化保存
            tmp_ckpt_path = latest_ckpt_path + ".tmp"
            json_file = f"training_history_split{split}.json"
            tmp_json_file = json_file + ".tmp"
            
            try:
                # 1. 安全保存 JSON 日志
                with open(tmp_json_file, "w", encoding="utf-8") as f:
                    json.dump(history_log, f, indent=4)
                os.replace(tmp_json_file, json_file) # os.replace 在 Linux 下是原子操作
                
                # 2. 安全保存 Checkpoint 权重
                torch.save(latest_state, tmp_ckpt_path)
                os.replace(tmp_ckpt_path, latest_ckpt_path)
                
                print(f"[*] 用于防崩溃的最新 checkpoint 已安全保存至 {latest_ckpt_path}")
                
            except Exception as e:
                print(f"\n[!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!]")
                print(f"[!] 警告: 保存 checkpoint 或日志失败 (大概率是磁盘空间已满)！")
                print(f"[!] 错误详情: {e}")
                print(f"[!] 已启动退化保护: 跳过当前 epoch ({epoch}) 的存档，上一次的完好存档已受保护。")
                print(f"[!] 模型将继续训练，请尽快清理服务器磁盘空间！")
                print(f"[!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!]\n")
                
                # 如果写了一半爆盘，顺手把残缺的 tmp 文件删掉，避免一直占着微薄的空间
                for tmp_file in [tmp_ckpt_path, tmp_json_file]:
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except Exception:
                            pass
            # =========================================================================

        # 修改：训练结束后如果有需要解锁的操作 (你原本的代码)
            
        # 修改：训练结束后如果有需要解锁的操作
        for key, value in dict(model.named_children()).items():
            if "proj" in key or "slowfast" in key or "aesthetic_conv3d" in key or "qformer" in key: # 加入 qformer
                for param in value.parameters():
                    param.requires_grad = True

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()