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
from functools import reduce
from thop import profile
import copy

from model.model import T2VQA
from dataset.dataset import T2VDataset

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            if len(line_split) == 3:
                filename, prompt, label = line_split
            elif len(line_split) == 4:
                filename, prompt, _, label = line_split
            else:
                raise ValueError(f"Invalid annotation line with {len(line_split)} fields: {line.strip()}")
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )

def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr

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

    # 适配 Qwen 的填空式 Prompt
    prompt_template = 'Carefully watch the video and evaluate its quality from the aspects of temporal consistency, aesthetic beauty, and semantic alignment. The overall quality of this video is'

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        inputs, video_up = {}, {} 

        # 适配重构后的模型输入：组装 inputs 字典
        inputs['video'] = data['video'].to(device)
        if "video_aesthetic" in data:
            inputs["video_aesthetic"] = data["video_aesthetic"].to(device)
        
        with torch.no_grad():
            caption = data['prompt']
            
            # 传入 inputs 字典和新的 prompt_template
            result["pr_labels"] = model(inputs, caption=caption, prompt=prompt_template).cpu().numpy()

            if len(list(video_up.keys())) > 0:
                result["pr_labels_up"] = model(video_up).cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del inputs, video_up
        results.append(result)

    ## generate the demo video for video quality localization
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
                # 适配新架构的参数名
                if (
                    "semantic_proj" in key 
                    or "motion_proj" in key 
                    or "aesthetic_proj" in key 
                    or "slowfast" in key 
                    or "aesthetic_conv3d" in key
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

    return best_s, best_p, best_k, best_r

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

    # 初始化重构后的模型
    model = T2VQA(opt["model"]["args"]).to(device)

    try:
        ckpt = torch.load(opt['test_load_path'], map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(opt['test_load_path'], map_location='cpu')

    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    # 清理旧架构残留权重（可选，防止不匹配报错）
    keys_to_remove = [k for k in state_dict.keys() if "finetune_Qformer" in k or "blip" in k or "swin" in k or "gate_mixer" in k]
    for k in keys_to_remove:
        if k in state_dict:
            del state_dict[k]
        
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights with msg: {msg}")
    model.eval()

    if opt.get("split_seed", -1) > 0:
        opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
        opt["data"]["eval"] = copy.deepcopy(opt["data"][args.target_set])

        split_duo = train_test_split(
            opt["data"][args.target_set]["args"]["data_prefix"],
            opt["data"][args.target_set]["args"]["anno_file"],
            seed=opt["split_seed"],
        )
        (
            opt["data"]["train"]["args"]["anno_file"],
            opt["data"]["eval"]["args"]["anno_file"],
        ) = split_duo

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

    bests = {}
    for key in val_loaders:
        bests[key] = -1, -1, -1, 1000

        bests[key] = inference_set(
            val_loaders[key],
            model,
            device,
            bests[key],
            save_model=False,
            save_name=None,
            suffix=key + "_s",
            save_type="full",
        )

    for key in val_loaders:
        print(
            f"""For the end-to-end transfer process on {key} with {len(val_loaders[key])} videos,
            the best validation accuracy of the model-s is as follows:
            SROCC: {bests[key][0]:.4f}
            PLCC:  {bests[key][1]:.4f}
            KROCC: {bests[key][2]:.4f}
            RMSE:  {bests[key][3]:.4f}."""
        )

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()