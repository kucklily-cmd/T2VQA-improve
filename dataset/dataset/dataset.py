import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import decord
from decord import VideoReader, cpu, gpu

from PIL import Image

decord.bridge.set_bridge("torch")

class SampleFrames:
    def __init__(self, clip_len, frame_interval=1, num_clips=1):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips

    def _get_train_clips(self, num_frames):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

        if avg_interval > 0:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = base_offsets + np.random.randint(
                avg_interval, size=self.num_clips
            )
        elif num_frames > max(self.num_clips, ori_clip_len):
            clip_offsets = np.sort(
                np.random.randint(num_frames - ori_clip_len + 1, size=self.num_clips)
            )
        elif avg_interval == 0:
            ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
            clip_offsets = np.around(np.arange(self.num_clips) * ratio)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        if train:
            clip_offsets = self._get_train_clips(total_frames)
        else:
            clip_offsets = self._get_test_clips(total_frames)
        frame_inds = (
            clip_offsets[:, None]
            + np.arange(self.clip_len)[None, :] * self.frame_interval
        )
        frame_inds = np.concatenate(frame_inds)

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        frame_inds = np.mod(frame_inds, total_frames)
        frame_inds = np.concatenate(frame_inds) + start_index
        return frame_inds.astype(np.int32)


class T2VDataset(Dataset):
    """Deformation of materials dataset."""

    def __init__(self, opt):
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.clip_len = opt["clip_len"]
        self.frame_interval = opt["frame_interval"]
        self.size = opt["size"]
        self.sampler = SampleFrames(self.clip_len, self.frame_interval)
        self.video_infos = []
        self.phase = opt["phase"]

        # ImageNet 标准化参数 (用于将 0-255 的 RGB 转换分布)
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split("|")
                    
                    # 修复 Bug: 兼容不同格式长度的标注文件
                    if len(line_split) == 3:
                        filename, prompt, label = line_split
                    elif len(line_split) == 4:
                        filename, prompt, _, label = line_split
                    else:
                        continue 
                        
                    label = float(label)
                    filename = os.path.join(self.data_prefix, filename)
                    self.video_infos.append(dict(filename=filename, prompt=prompt, label=label))

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        prompt = video_info["prompt"]
        label = video_info["label"]
        
        vreader = VideoReader(filename)
        
        frame_inds = self.sampler(len(vreader), self.phase == "train")
        frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}

        imgs = [frame_dict[idx] for idx in frame_inds]
        img_shape = imgs[0].shape # [H, W, C]
        
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2) # [C, T, H, W]
        
        # ==========================================================
        # 1. 基础视觉分支处理 (用于 CLIP 和 SlowFast)
        # ==========================================================
        # 直接暴力缩放到 224x224 (可能会导致拉伸变形，但保留了画面边缘的所有信息)
        video_base = torch.nn.functional.interpolate(video, size=(self.size, self.size))
        vfrag_base = ((video_base.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        # ==========================================================
        # 2. 核心适配：美学分支专用视频 (保持宽高比的裁剪)
        # ==========================================================
        # 美学特征极度依赖未被破坏的原始画面比例，因此采用 CenterCrop 保持画面比例不失真
        H, W = video.shape[2], video.shape[3]
        min_dim = min(H, W)
        crop_h = (H - min_dim) // 2
        crop_w = (W - min_dim) // 2
        
        # 截取中心正方形区域，然后缩放
        video_aes = video[:, :, crop_h : crop_h + min_dim, crop_w : crop_w + min_dim]
        video_aes = torch.nn.functional.interpolate(video_aes, size=(self.size, self.size))
        vfrag_aes = ((video_aes.permute(1, 2, 3, 0) - self.mean) / self.std).permute(3, 0, 1, 2)

        data = {
            "video": vfrag_base,             # 供 CLIP 和 SlowFast 使用
            "video_aesthetic": vfrag_aes,    # 供 ConvNeXt(美学分支) 独享使用
            "prompt": prompt,
            "frame_inds": frame_inds,
            "gt_label": label,
            "original_shape": img_shape,
        }
        
        return data