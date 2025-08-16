# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset

# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None):
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in
                           range(start_frame, start_frame + sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)

            random_y = []

            full_range = np.arange(0, num_frames - sample_duration + 1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame - sample_duration, start_frame + sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in
                           range(random_start_frame, random_start_frame + sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return (y, random_y)

        else:  # without contrastive learning
            y = []
            if self.shuffle_time_sequence:  # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in
                               random.sample(list(range(0, num_frames)), sample_duration // self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in
                               range(start_frame, start_frame + sample_duration, self.stride_within_seq)]

            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path,weights_only=False).unsqueeze(0)
                y_i = y_i.unsqueeze(-1)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class TripleDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        dataset_labels = {
            "CN": 0,
            "PD": 1,
            "Prodromal": 2
        }

        # 遍历三个数据集
        for dataset_name, label in dataset_labels.items():
            # 修正数据集路径 - 直接使用数据集目录
            dataset_path = os.path.join(root, f"{dataset_name}_MNI_to_TRs_minmax")

            # 检查是否存在img子目录
            img_path = os.path.join(dataset_path, "img")
            if os.path.exists(img_path):
                dataset_path = img_path  # 使用img子目录

            if not os.path.exists(dataset_path):
                print(f"警告: 数据集路径不存在 - {dataset_path}")
                continue

            # 遍历每个受试者目录
            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path):
                    print(f"跳过非目录项: {subject_path}")
                    continue

                # 检查是否是有效受试者目录 (包含frame文件)
                frame_files = [f for f in os.listdir(subject_path)
                               if f.startswith('frame_') and f.endswith('.pt')]
                if not frame_files:
                    print(f"警告: 受试者没有帧数据 - {subject_path}")
                    continue

                num_frames = len(frame_files)

                # 确保序列长度有效
                session_duration = num_frames - self.sample_duration + 1
                if session_duration <= 0:
                    print(f"警告: 序列长度不足 - {subject_path} (需要 {self.sample_duration} 帧，实际 {num_frames} 帧)")
                    continue

                # 为每个有效序列创建数据项
                for start_frame in range(0, session_duration, self.stride):
                    unique_subject = f"{dataset_name}_{subject}"
                    # 检查受试者是否在 subject_dict 中
                    if unique_subject in subject_dict:
                        data_tuple = (len(data), unique_subject, subject_path,
                                      start_frame, num_frames, label)
                        data.append(data_tuple)

        # 打印数据集统计信息
        print(f"已加载数据集: {len(data)} 个序列")
        if self.train:
            self.target_values = np.array([tup[5] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, num_frames, target= self.data[index]

        if self.contrastive:
            # 对比学习模式
            y, rand_y = self.load_sequence(subject_path, start_frame, self.sample_duration)
            y = self._pad_to_size(y)
            rand_y = self._pad_to_size(rand_y)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
            }
        else:
            # 标准模式
            y = self.load_sequence(subject_path, start_frame, self.sample_duration, num_frames)
            y = self._pad_to_size(y)
            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
            }

    def _pad_to_size(self, tensor):
        """将fMRI张量动态填充到96x96x96尺寸"""
        _, d, h, w, t = tensor.shape
        background_value = tensor.flatten()[0]
        tensor = tensor.permute(0, 4, 1, 2, 3)  # 1 20 61 73 61
        # tensor = torch.nn.functional.pad(tensor, (18, 17, 12, 11, 18, 17), value=background_value)
        tensor = torch.nn.functional.pad(tensor, (10, 9, 4, 3, 10, 9), value=background_value)
        tensor = tensor.permute(0, 2, 3, 4, 1)

        return tensor


class FA3DDataset(BaseDataset):
    """3D FA值数据集加载器，与原始TripleDataset结构一致"""

    def __init__(self, root, subject_dict, sample_duration=1, stride=1, train=True, **kwargs):
        """
        Args:
            root (str): 数据集根目录
            subject_dict (dict): 受试者ID到索引的映射
            sample_duration (int): 每个样本的帧数（对于3D数据通常为1）
            stride (int): 采样步长
            train (bool): 是否为训练集
        """
        self.root = root
        self.subject_dict = subject_dict
        self.sample_duration = sample_duration
        self.stride = stride
        self.train = train

        # 设置数据集标签
        self.dataset_labels = {
            "CN": 0,
            "PD": 1,
            "Prodromal": 2
        }

        # 加载数据路径
        self.data = self._set_data()

        # 打印数据集统计信息
        print(f"已加载数据集: {len(self.data)} 个序列")
        if self.train:
            self.target_values = np.array([tup[5] for tup in self.data]).reshape(-1, 1)

    def _set_data(self):
        """加载所有有效数据路径 - 适配3D FA数据格式"""
        data = []

        # 遍历三个数据集
        for dataset_name, label in self.dataset_labels.items():
            # 修正数据集路径
            dataset_path = os.path.join(self.root, f"{dataset_name}_FA_processed", "img")

            if not os.path.exists(dataset_path):
                print(f"警告: 数据集路径不存在 - {dataset_path}")
                continue

            # 遍历每个受试者目录
            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path):
                    print(f"跳过非目录项: {subject_path}")
                    continue

                # 检查是否是有效受试者目录 (包含FA文件)
                fa_path = os.path.join(subject_path, "fa.pt")
                if not os.path.exists(fa_path):
                    print(f"警告: 受试者没有FA数据 - {subject_path}")
                    continue

                # 对于FA数据，我们只有一个文件，所以只创建一个数据项
                unique_subject = f"{dataset_name}_{subject}"

                # 检查受试者是否在 subject_dict 中
                if unique_subject in self.subject_dict:
                    # 对于FA数据，我们使用固定的 start_frame=0 和 sample_duration=1
                    data_tuple = (
                        len(data),
                        unique_subject,
                        subject_path,  # 存储目录路径
                        0,  # start_frame 总是0
                        1,  # 帧数总是1 (只有1个FA文件)
                        label
                    )
                    data.append(data_tuple)

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, num_frames, target = self.data[index]

        # 加载3D FA数据 - 与原始代码加载方式一致
        fa_data = self.load_sequence(subject_path, start_frame, self.sample_duration, num_frames)

        # 动态填充到标准尺寸 (与原始_pad_to_size方法一致)
        fa_data = self._pad_to_size(fa_data)
        fa_data = fa_data.float()
        return {
            "dti_sequence": fa_data,
            "subject_name": subject,
            "target": target,
            "TR": start_frame
        }

    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames):
        """加载FA数据 - 适配单个3D FA张量文件"""
        # 对于FA数据，我们只有一个文件 'fa.pt'，而不是多个帧文件
        fa_path = os.path.join(subject_path, 'fa.pt')

        # 检查文件是否存在
        if not os.path.exists(fa_path):
            raise FileNotFoundError(f"FA文件不存在: {fa_path}")

        # 加载3D FA数据
        fa_tensor = torch.load(fa_path, weights_only=True)

        # 确保正确的维度
        if fa_tensor.dim() == 3:  # (D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加通道维度 -> (1, D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加时间维度 -> (1, 1, D, H, W)
        elif fa_tensor.dim() == 4:  # (C, D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加时间维度 -> (1, C, D, H, W)
        elif fa_tensor.dim() == 5:  # (T, C, D, H, W)
            pass  # 已经是正确格式
        else:
            raise ValueError(f"无效的FA数据维度: {fa_tensor.shape}，路径: {fa_path}")

        # 对于FA数据，我们只有一个时间点，所以直接返回
        return fa_tensor

    def _pad_to_size(self, tensor):
        # 处理不同输入形状:
        if tensor.dim() == 4:  # (C, D, H, W)
            c, d, h, w = tensor.shape
            t = 1
            tensor = tensor.unsqueeze(0)  # 添加时间维度 (1, C, D, H, W)
        elif tensor.dim() == 5:  # (T, C, D, H, W)
            t, c, d, h, w = tensor.shape
        else:
            raise ValueError(f"不支持的FA张量维度: {tensor.dim()}")

        # 获取背景值 (使用图像角落的值)
        background_value = tensor[0, 0, 0, 0, 0].item()

        # 计算填充参数 - 与原始填充参数一致
        # 原始填充: (18, 17, 12, 11, 18, 17) 对应 (W左, W右, H上, H下, D前, D后)
        padding = (10, 9, 4, 3, 10, 9) #61 73 61 80 80 80

        # 应用填充 (PyTorch的填充顺序: W, H, D)
        padded_tensor = torch.nn.functional.pad(
            tensor,
            padding,
            mode='constant',
            value=background_value
        )

        # 恢复原始形状
        if t == 1:
            padded_tensor = padded_tensor.squeeze(0)

        return padded_tensor


class MultimodalDataset(BaseDataset):
    def __init__(self, fmri_root, fa_root, subject_dict, sequence_length=20, stride_within_seq=1,
                 stride_between_seq=0.5, train=True, contrastive=False, with_voxel_norm=False,
                 shuffle_time_sequence=False, **kwargs):
        """
        重构版多模态数据集 - 每个序列视为独立样本

        参数:
            fmri_root: fMRI数据根目录
            fa_root: FA数据根目录
            subject_dict: 受试者字典
            sequence_length: fMRI序列长度
            stride_within_seq: fMRI序列内步长
            stride_between_seq: fMRI序列间步长
            train: 是否为训练集
            contrastive: 是否用于对比学习
            with_voxel_norm: 是否使用体素归一化
            shuffle_time_sequence: 是否打乱时间序列
        """
        self.fmri_root = fmri_root
        self.fa_root = fa_root
        self.subject_dict = subject_dict or {}
        self.sequence_length = sequence_length
        self.stride_within_seq = stride_within_seq
        self.stride_between_seq = stride_between_seq
        self.train = train
        self.contrastive = contrastive
        self.with_voxel_norm = with_voxel_norm
        self.shuffle_time_sequence = shuffle_time_sequence

        # 计算序列参数
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration), 1)

        self.dataset_labels = {
            "CN": 0,
            "PD": 1,
            "Prodromal": 2
        }

        # 加载序列样本数据
        self.sequence_samples = self._load_sequence_samples()

        # 打印数据集统计信息
        num_sequences = len(self.sequence_samples)
        num_subjects = len(self.subject_dict)
        print(f"已加载多模态数据集: {num_sequences} 个序列样本")


    def _load_sequence_samples(self):
        """加载所有序列样本（每个序列视为独立样本）"""
        sequence_samples = []

        # 如果没有受试者，直接返回空列表
        if not self.subject_dict:
            return sequence_samples

        # 扫描所有受试者
        for unique_subject in self.subject_dict:
            # 解析数据集和受试者ID
            parts = unique_subject.split("_", 1)
            if len(parts) < 2:
                print(f"警告: 无效的受试者ID格式 - {unique_subject}")
                continue

            dataset_name, subject_id = parts
            label = self.subject_dict[unique_subject][1]

            # 构建数据路径
            fmri_path = os.path.join(self.fmri_root, f"{dataset_name}_MNI_to_TRs_minmax", subject_id)
            fa_path = os.path.join(self.fa_root, f"{dataset_name}_FA_processed", "img", subject_id)

            # 检查路径是否存在
            if not os.path.exists(fmri_path):
                print(f"警告: fMRI路径不存在 - {fmri_path}")
                continue
            if not os.path.exists(fa_path):
                print(f"警告: FA路径不存在 - {fa_path}")
                continue

            # 获取可用时间帧数
            try:
                fmri_files = [f for f in os.listdir(fmri_path) if f.startswith('frame_') and f.endswith('.pt')]
                num_frames = len(fmri_files)
            except Exception as e:
                print(f"读取fMRI文件列表错误: {fmri_path}, {str(e)}")
                continue

            # 计算可用序列位置
            session_duration = num_frames - self.sample_duration + 1
            if session_duration <= 0:
                print(f"跳过受试者 {unique_subject}，序列长度不足 (需要 {self.sample_duration} 帧，实际 {num_frames} 帧)")
                continue

            # 为该受试者创建所有可能的序列样本
            for start_frame in range(0, session_duration, self.stride):
                sequence_samples.append({
                    "unique_subject": unique_subject,
                    "dataset_name": dataset_name,
                    "subject_id": subject_id,
                    "fmri_path": fmri_path,
                    "fa_path": fa_path,
                    "start_frame": start_frame,
                    "label": label,
                    "num_frames": num_frames
                })

        return sequence_samples

    def __len__(self):
        """返回序列样本总数"""
        return len(self.sequence_samples)

    def _load_fmri_sequence(self, subject_path, start_frame, sample_duration, num_frames):
        """加载fMRI序列"""
        if self.contrastive:
            # 对比学习模式
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in
                           range(start_frame, start_frame + sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)

            random_y = []
            full_range = np.arange(0, num_frames - sample_duration + 1)
            # 计算排除范围（避免相邻序列）
            exclude_range = np.arange(
                max(0, start_frame - sample_duration),
                min(num_frames, start_frame + sample_duration)
            )
            available_choices = np.setdiff1d(full_range, exclude_range)

            if len(available_choices) > 0:
                random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            else:
                random_start_frame = 0  # 回退方案

            load_fnames = [f'frame_{frame}.pt' for frame in
                           range(random_start_frame, random_start_frame + sample_duration, self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return y, random_y
        else:
            # 标准模式
            y = []
            if self.shuffle_time_sequence:
                # 随机选择时间点，不保持连续性
                frames = random.sample(range(num_frames), self.sequence_length)
                load_fnames = [f'frame_{frame}.pt' for frame in frames]
            else:
                # 连续序列
                load_fnames = [f'frame_{frame}.pt' for frame in
                               range(start_frame, start_frame + sample_duration, self.stride_within_seq)]

            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path, weights_only=False).unsqueeze(0)
                y_i = y_i.unsqueeze(-1)  # 添加时间维度
                y.append(y_i)
            return torch.cat(y, dim=4)

    def _load_fa_data(self, subject_path):
        """加载FA数据"""
        fa_path = os.path.join(subject_path, 'fa.pt')
        if not os.path.exists(fa_path):
            raise FileNotFoundError(f"FA文件不存在: {fa_path}")

        fa_tensor = torch.load(fa_path, weights_only=False)

        if fa_tensor.dim() == 3:  # (D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加通道维度 -> (1, D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加时间维度 -> (1, 1, D, H, W)
        elif fa_tensor.dim() == 4:  # (C, D, H, W)
            fa_tensor = fa_tensor.unsqueeze(0)  # 添加时间维度 -> (1, C, D, H, W)
        elif fa_tensor.dim() == 5:  # (T, C, D, H, W)
            pass  # 已经是正确格式

        return fa_tensor

    def _pad_fmri(self, tensor):
        """填充fMRI张量到标准尺寸"""
        _, d, h, w, t = tensor.shape
        background_value = tensor.flatten()[0]
        tensor = tensor.permute(0, 4, 1, 2, 3)  # [C, T, D, H, W]
        tensor = torch.nn.functional.pad(tensor, (10, 9, 4, 3, 10, 9), value=background_value)
        return tensor.permute(0, 2, 3, 4, 1)  # 恢复为 [C, D, H, W, T]

    def _pad_fa(self, tensor):
        """填充FA张量到标准尺寸"""
        # 确保维度正确
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        elif tensor.dim() == 4:
            tensor = tensor.unsqueeze(0)  # [1,C,D,H,W] -> [1,1,C,D,H,W]?
        elif tensor.dim() != 5:
            raise ValueError(f"无效的FA数据维度: {tensor.shape}")

        t, c, d, h, w = tensor.shape
        background_value = tensor[0, 0, 0, 0, 0].item()
        padding = (10, 9, 4, 3, 10, 9)  # 左右、上下、前后填充
        padded_tensor = torch.nn.functional.pad(tensor, padding, value=background_value)
        # 恢复原始形状
        if t == 1:
            padded_tensor = padded_tensor.squeeze(0)
        return padded_tensor

    def __getitem__(self, index):
        """获取指定索引的序列样本"""
        sample = self.sequence_samples[index]

        # 加载fMRI序列
        if self.contrastive:
            fmri_seq, fmri_rand_seq = self._load_fmri_sequence(
                sample["fmri_path"],
                sample["start_frame"],
                self.sample_duration,
                sample["num_frames"]
            )
            fmri_seq = self._pad_fmri(fmri_seq)
            fmri_rand_seq = self._pad_fmri(fmri_rand_seq)
        else:
            fmri_seq = self._load_fmri_sequence(
                sample["fmri_path"],
                sample["start_frame"],
                self.sample_duration,
                sample["num_frames"]
            )
            fmri_seq = self._pad_fmri(fmri_seq)

        # 加载FA数据
        fa_data = self._load_fa_data(sample["fa_path"])
        fa_data = self._pad_fa(fa_data)

        # 返回样本数据
        result = {
            "fmri_sequence": fmri_seq,
            "fa_sequence": fa_data,
            "subject_name": sample["unique_subject"],
            "target": sample["label"],
            "TR": sample["start_frame"],
        }

        # 对比学习模式返回额外数据
        if self.contrastive:
            result["fmri_contrastive"] = fmri_rand_seq

        return result