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
            label = float(label)# 得分
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],# 训练集
        video_infos[int(ratio * len(video_infos)) :],# 测试集
    )

def plcc_loss(y_pred, y):
    # 先计算方差，加上平滑项后再开方，保证反向传播时梯度安全
    var_pred, m_hat = torch.var_mean(y_pred, unbiased=False)
    sigma_hat = torch.sqrt(var_pred + 1e-8)
    y_pred = (y_pred - m_hat) / sigma_hat
    
    var_y, m = torch.var_mean(y, unbiased=False)
    sigma = torch.sqrt(var_y + 1e-8)
    y = (y - m) / sigma
    
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rank_loss(y_pred, y):
    # 强制转为列向量
    y_pred = y_pred.view(-1, 1)
    y = y.view(-1, 1)
    
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

# 待看
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
    total_loss = 0.0  # 新增：用于累加当前 epoch 的总 loss
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        video["video"] = data["video"].to(device)
        video["frame_inds"] = data["frame_inds"].to(device)

        y = data["gt_label"].float().detach().to(device)

        captions = data['prompt']
        prompts = [
            f"The user provided the text prompt: '{c}' to generate a video. Please assess the overall quality of this generated video, considering both text-video alignment and visual fidelity." 
            for c in captions
        ]
        scores = model(video, caption=captions, prompt=prompts)

        y_pred = scores
        # if len(scores) > 1:
        #     y_pred = reduce(lambda x, y: x + y, scores)
        # else:
        #     y_pred = scores[0]
        # y_pred = y_pred.mean((-3, -2, -1))

        p_loss = plcc_loss(y_pred, y)
        r_loss = rank_loss(y_pred, y)

        loss = p_loss + 0.3 * r_loss

        loss.backward()
        # 【新增】梯度裁剪，防止 Transformer 梯度异常放大
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # ft_loader.dataset.refresh_hypers()
        total_loss += loss.item()  # 新增：累加 loss

    model.eval()
    avg_loss = total_loss / len(ft_loader)  # 新增：计算平均 loss
    return avg_loss  # 新增：返回当前轮次的平均 loss

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

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video, video_up = {}, {}

        video['video'] = data['video'].to(device)
        
        ## Reshape into clips
        b, c, t, h, w = video['video'].shape
            
        with torch.no_grad():
            # 原代码：
            # prompt = 'Please assess the quality of this video'
            # caption = data['prompt']
            # result["pr_labels"] = model(video, caption = caption, prompt = prompt).cpu().numpy()

            # 替换为：
            captions = data['prompt']
            prompts = [
                f"The user provided the text prompt: '{c}' to generate a video. Please assess the overall quality of this generated video, considering both text-video alignment and visual fidelity." 
                for c in captions
            ]
            result["pr_labels"] = model(video, caption=captions, prompt=prompts).cpu().numpy()

            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del video, video_up
        results.append(result)

    ## generate the demo video for video quality localization
    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    # wandb.log(
    #     {
    #         f"val_{suffix}/SRCC-{suffix}": s,
    #         f"val_{suffix}/PLCC-{suffix}": p,
    #         f"val_{suffix}/KRCC-{suffix}": k,
    #         f"val_{suffix}/RMSE-{suffix}": r,
    #     }
    # )

    del results, result  # , video, video_up
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                # 修改为：
                if (
                    "finetune" in key
                    or "swin" in key
                    or "conv" in key
                    or "gate_mixer" in key
                    or "slowfast" in key
                    or "blip.text_encoder" in key
                    or "lora" in key  # 【新增】这一行，确保视觉编码器的 lora 权重落盘保存
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
    # 参数解析器
    # 解析命令行传入参数格式 python train.py --opt my_config.yml
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--opt", type=str, default="t2vqa.yml", help="the option file"# 配置文件
    )

    parser.add_argument(
        "-t", "--target_set", type=str, default="t2v", help="target_set"# 数据集
    )

    args = parser.parse_args()
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ## defining model and loading checkpoint

    bests_ = []
    # 找不到这个字段返回-1
    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1

    print(opt["split_seed"])

    for split in range(num_splits):
        model = T2VQA(opt["model"]["args"]).to(device)# 把模型方式GPU
        if opt.get("split_seed", -1) > 0:
            # 配置文件的结构应该是data target 还有T2VQA 指明它的一些数据集配置内容
            # 数据集划分逻辑
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

        # 利用参数组实现部分更新的策略
        # 利用参数组实现部分更新的策略，赋予不同模块不同的学习率
        param_groups = []   

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # 1. 如果是 LoRA 的参数，给予较大的学习率 (例如 1e-4)
            if "lora" in name:
                param_groups.append({
                    "params": param, 
                    "lr": 1e-4  # LoRA 专用学习率
                })
            # 2. 如果是视觉编码器、门控融合等其他可训练参数，使用配置文件中的基础学习率 (1e-5)
            elif (
                "finetune" in name
                or "swin" in name
                or "conv3d" in name
                or "gate_mixer" in name
                or "slowfast" in name
                or "blip.text_encoder" in name
            ):
                param_groups.append({
                    "params": param, 
                    "lr": opt["optimizer"]["lr"]  # 读取 yml 里的 1e-5
                })


        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )

        #训练轮数计算 
        warmup_iter = 0 
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"]) * len(train_loader))

        #线性热身，余弦退火，学习率策略
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        #控制哪一个优化器，自定义学习率策略
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda,)

        # python元组存储四个不同的指标
        bests = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
        history_log = []  # 新增：初始化用于记录所有轮次数据的列表
        for epoch in range(opt["num_epochs"]):
            print(f"End-to-end Epoch {epoch}:")
            epoch_train_loss = 0.0  
            
            for key, train_loader in train_loaders.items():
                # 接收当前 epoch 的训练 loss
                epoch_train_loss = finetune_epoch(
                    train_loader,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    epoch,
                )
            
            # 初始化当前 epoch 的数据字典
            epoch_data = {
                "epoch": epoch,
                "train_loss": epoch_train_loss
            }

            for key in val_loaders:
                # 拆包接收历史最佳指标和当前指标
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
                
                # 将当前轮次的各项指标提取到字典中
                s, p, k, r = current_metrics
                epoch_data[f"val_{key}_SRCC"] = float(s)
                epoch_data[f"val_{key}_PLCC"] = float(p)
                epoch_data[f"val_{key}_KRCC"] = float(k)
                epoch_data[f"val_{key}_RMSE"] = float(r)
            
            # 将当前轮次数据加入总记录中
            history_log.append(epoch_data)
            
            # 每一轮结束将纯数据落盘保存（覆盖更新），无需保存模型权重
            with open(f"training_history_split{split}.json", "w", encoding="utf-8") as f:
                json.dump(history_log, f, indent=4)
            

        # 精准解锁模型中的特定子模块，允许它们在训练过程中更新参数。
        for key, value in dict(model.named_children()).items():
            if "finetune" in key or "swin" in key or "conv" in key or "gate_mixer" in key or "slowfast" in key:
                for param in value.parameters():
                    param.requires_grad = True

        #清理结尾
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
