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

from model.model import T2VQA_Llama3_Blip2 as T2VQA # 引用新写的类名
from dataset.dataset import T2VDataset

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(f"Split seed: {seed}")
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
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
    scaler, # 传入 AMP 梯度缩放器
    epoch=-1,
):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        optimizer.zero_grad()
        video = {}
        video["video"] = data["video"].to(device)
        if "frame_inds" in data:
            video["frame_inds"] = data["frame_inds"].to(device)

        y = data["gt_label"].float().detach().to(device)
        caption = data['prompt']
        prompt = 'Please assess the quality of this video'

        # 【更新】：使用新版 PyTorch 混合精度 API，消除 FutureWarning
        with torch.amp.autocast('cuda'):
            scores = model(video, caption=caption, prompt=prompt)
            y_pred = scores

            p_loss = plcc_loss(y_pred, y)
            r_loss = rank_loss(y_pred, y)
            loss = p_loss + 0.3 * r_loss

        # 使用 scaler 进行反向传播和优化
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
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

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        video = {}
        video['video'] = data['video'].to(device)
            
        with torch.no_grad():
            prompt = 'Please assess the quality of this video'
            caption = data['prompt']
            
            # 【更新】：推理时同样使用新版混合精度 API
            with torch.amp.autocast('cuda'):
                preds = model(video, caption=caption, prompt=prompt)
            
            result["pr_labels"] = preds.cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        results.append(result)

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                # 剔除 LLaMA-3 的未参与训练的基础参数
                if "llm_model.base_model" in key and "lora" not in key:
                    continue
                # 【极其重要】：剔除 BLIP-2 基础参数，但保留 Vision LoRA 权重！
                if "blip2.vision_model" in key and "lora" not in key:
                    continue
                
                head_state_dict[key] = v.cpu() # 移至 cpu 节省保存内存
                
            print("Saving Checkpoint... Keys included: ", len(head_state_dict.keys()))
            torch.save(
                {"state_dict": head_state_dict, "validation_results": best_},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )
        else:
            torch.save(
                {"state_dict": state_dict, "validation_results": best_},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )

    best_s, best_p, best_k, best_r = (
        max(best_s, s), max(best_p, p), max(best_k, k), min(best_r, r),
    )

    print(
        f"For {len(inf_loader)} videos, \nthe accuracy of the model: [{suffix}] is as follows:\n  SROCC: {s:.4f} best: {best_s:.4f} \n  PLCC:  {p:.4f} best: {best_p:.4f}  \n  KROCC: {k:.4f} best: {best_k:.4f} \n  RMSE:  {r:.4f} best: {best_r:.4f}."
    )

    return (best_s, best_p, best_k, best_r), (s, p, k, r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="t2vqa.yml", help="the option file")
    parser.add_argument("-t", "--target_set", type=str, default="t2v", help="target_set")
    args = parser.parse_args()
    
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print("Config Loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bests_ = []
    num_splits = 10 if opt.get("split_seed", -1) > 0 else 1

    for split in range(num_splits):
        print(f"--- Starting Split {split} ---")
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
                train_dataset = T2VDataset(opt["data"][key]["args"])
                train_datasets[key] = train_dataset

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
                val_dataset = T2VDataset(opt["data"][key]["args"])
                val_datasets[key] = val_dataset

        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=1,
                num_workers=opt["num_workers"],
                pin_memory=True,
            )

        # 自动筛选所有需要求导的参数（包括 LLM LoRA、Vision LoRA、Q-Former、Swin等）
        param_groups = []   
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_groups.append({"params": param, "lr": opt["optimizer"]["lr"]})

        # 打印一下实际参与训练的参数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {trainable_params / 1e6:.2f} M")

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
        
        # 【更新】：使用新版 API 创建 AMP 梯度缩放器，消除 FutureWarning
        scaler = torch.amp.GradScaler('cuda')

        bests = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
            
        history_log = []
        for epoch in range(opt["num_epochs"]):
            print(f"End-to-end Epoch {epoch}:")
            epoch_train_loss = 0.0  
            
            for key, train_loader in train_loaders.items():
                epoch_train_loss = finetune_epoch(
                    train_loader,
                    model,
                    optimizer,
                    scheduler,
                    device,
                    scaler, # 传入 scaler
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
                    save_model=opt.get("save_model", True),
                    save_name=opt.get("wandb", {}).get("project_name", "T2VQA") + "_head_" + args.target_set + f"_{split}",
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

        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()