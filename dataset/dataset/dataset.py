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
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
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
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
        return clip_offsets

    def _get_test_clips(self, num_frames, start_index=0):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int32)
        else:
            clip_offsets = np.zeros((self.num_clips,), dtype=np.int32)
        return clip_offsets

    def __call__(self, total_frames, train=False, start_index=0):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
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
        # 视频文件存放的根目录前缀
        self.data_prefix = opt["data_prefix"]
        self.clip_len = opt["clip_len"]
        self.frame_interval = opt["frame_interval"]
        self.size = opt["size"]
        # 抽帧器
        self.sampler = SampleFrames(self.clip_len, self.frame_interval)
        self.video_infos = []
        # 当前阶段，区分训练测试和验证
        self.phase = opt["phase"]

        #图像标准化参数，使用它们能让模型输入的数据分布在 0 附近，从而让训练更稳定、收敛更快。
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])

        if isinstance(self.ann_file, list):
            self.video_infos = self.ann_file
        else:
            with open(self.ann_file, "r") as fin:
                for line in fin:
                    line_split = line.strip().split("|")
                    filename, prompt, label = line_split
                    label = float(label)
                    filename = os.path.join(self.data_prefix, filename)
                    # 结构化数据为字典
                    self.video_infos.append(dict(filename=filename, prompt=prompt, label=label))

    def __len__(self):
        
        return len(self.video_infos)

    def __getitem__(self, index):
        video_info = self.video_infos[index]
        filename = video_info["filename"]
        prompt = video_info["prompt"]
        label = video_info["label"]
        
        vreader = VideoReader(filename)
        
        # 此时 sampler 应该根据 t2vqa.yml 的 clip_len=32 采样 32 帧
        frame_inds = self.sampler(len(vreader), self.phase == "train")
        frame_dict = {idx: vreader[idx] for idx in np.unique(frame_inds)}

        imgs = [frame_dict[idx] for idx in frame_inds]
        img_shape = imgs[0].shape
        
        # 将三维度图片列表，沿着时间轴变成，四维度的视频向量 [T, H, W, C]
        video = torch.stack(imgs, 0)
        # 维度重排变为 [C, T, H, W]，并转换为 float 类型（这一步非常重要，否则后面的相减计算会报错或溢出）
        video = video.permute(3, 0, 1, 2).float() 
        # 空间缩放为 224×224，此时 video_tensor 的维度是 [C, 32, 224, 224]
        video_tensor = torch.nn.functional.interpolate(video, size=(self.size, self.size))

        # ==================== 1. 语义一致性分支 (全局 + 低帧率) ====================
        # 我们从 32 帧中均匀抽取 8 帧，保留 224x224 的完整空间视野
        num_semantic_frames = 8
        T = video_tensor.shape[1]
        if T >= num_semantic_frames:
            interval = T // num_semantic_frames
            # 加上前缀索引抽取 8 帧
            video_semantic = video_tensor[:, ::interval, :, :][:, :num_semantic_frames, :, :]
        else:
            video_semantic = video_tensor

        # ==================== 2. 运动保真度分支 (Fragment + 高帧率) ====================
        # 强烈建议使用 128 而不是 112，因为 128 能被 32 整除，防止 ConvNext 下采样报错
        crop_size = 128  
        _, _, H, W = video_tensor.shape

        # 计算整段视频的“运动显著性” (通过相邻帧的绝对差值来衡量运动幅度)
        if T > 1:
            # 相邻帧相减并取绝对值，然后在通道和时间维度求平均，得到 [H, W] 的运动热图
            motion_map = torch.mean(torch.abs(video_tensor[:, 1:, :, :] - video_tensor[:, :-1, :, :]), dim=(0, 1))
        else:
            motion_map = torch.mean(video_tensor, dim=(0, 1))

        # 定义 5 个候选 Fragment 区域 (四个角 + 中心)
        candidates = [
            (0, 0), (0, W - crop_size), 
            (H - crop_size, 0), (H - crop_size, W - crop_size),
            ((H - crop_size) // 2, (W - crop_size) // 2)
        ]

        # 找到运动得分最高的一个 Patch
        best_score = -1
        best_h, best_w = 0, 0
        for (h_start, w_start) in candidates:
            score = torch.mean(motion_map[h_start:h_start+crop_size, w_start:w_start+crop_size]).item()
            if score > best_score:
                best_score = score
                best_h, best_w = h_start, w_start

        # 截取最具运动显著性的所有帧的 128x128 视频块 -> [C, 32, 128, 128]
        video_fidelity = video_tensor[:, :, best_h:best_h+crop_size, best_w:best_w+crop_size]

        # ==================== 组装返回数据 ====================
        data = {
            'video_fidelity': video_fidelity,  # [C, 32, 128, 128]
            'video_semantic': video_semantic,  # [C, 8, 224, 224]
            'frame_inds': frame_inds,
            'prompt': prompt,  # 修改为你头部定义的变量名 prompt
            'gt_label': label,
            "original_shape": img_shape,
        }
        
        return data