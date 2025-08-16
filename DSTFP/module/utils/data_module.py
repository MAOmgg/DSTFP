import os
from collections import defaultdict

import pytorch_lightning as pl

from torch.utils.data import DataLoader
from .data_preprocess_and_load.datasets import TripleDataset, FA3DDataset, MultimodalDataset
from argparse import ArgumentParser
from .parser import str2bool

from sklearn.model_selection import StratifiedShuffleSplit

import os
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)


class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # generate splits folder
        if self.hparams.pretraining:
                split_dir_path = f'./Splits/{self.hparams.dataset_name}/pretraining'
        else:
            split_dir_path = f'./Splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
        
        self.setup()

        #pl.seed_everything(seed=self.hparams.data_seed)

    def get_dataset(self):
        if self.hparams.dataset_name == "Triple":
            return TripleDataset
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        #subj_idx = np.array([str(x[0]) for x in subj_list])
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        # print(S)
        print('unique subjects:',len(S))  
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx
    
    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def determine_split_randomly(self, S):
        """划分训练集和测试集，验证集设为空"""
        subjects = list(S.keys())
        labels = [S[subject][1] for subject in subjects]

        # 单次分层划分：训练集 vs 测试集
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - self.hparams.train_split,
            random_state=self.hparams.data_seed
        )

        train_idx, test_idx = next(sss.split(subjects, labels))
        S_train = [subjects[i] for i in train_idx]
        S_test = [subjects[i] for i in test_idx]
        S_val = []


        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test

    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return
    
    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()
        # 添加TripleDataset的处理
        if self.hparams.dataset_name == "Triple":
            # 定义数据集名称到标签的映射
            dataset_labels = {
                "CN": 0,
                "PD": 1,
                "Prodromal": 2
            }

            # 遍历三个数据集目录
            for dataset_name, label in dataset_labels.items():
                dataset_path = os.path.join(self.hparams.image_path, f"{dataset_name}_MNI_to_TRs_minmax")
                if not os.path.exists(dataset_path):
                    print(f"警告: 数据集路径不存在 - {dataset_path}")
                    continue

                # 获取该数据集下的所有受试者
                subjects = [d for d in os.listdir(dataset_path)
                            if os.path.isdir(os.path.join(dataset_path, d))]

                for subject in subjects:
                    # 使用数据集前缀确保受试者名称唯一
                    unique_subject = f"{dataset_name}_{subject}"
                    # 标签直接使用数据集定义的标签，性别设为0（不需要）
                    final_dict[unique_subject] = [0, label]
        else:
            raise NotImplementedError

        return final_dict

    def setup(self, stage=None):
        # this function will be called at each devices

        Dataset = self.get_dataset()
        params = {
                "root": self.hparams.image_path,
                "sequence_length": self.hparams.sequence_length,
                "contrastive":self.hparams.use_contrastive,
                "contrastive_type":self.hparams.contrastive_type,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task": self.hparams.downstream_task,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "input_type": self.hparams.input_type,
                "label_scaling_method" : self.hparams.label_scaling_method,
                "dtype":'float16'}
        
        subject_dict = self.make_subject_dict()
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)
        
        if self.hparams.bad_subj_path:
            bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
            for bad_subj in bad_subjects:
                bad_subj = bad_subj.strip()
                if bad_subj in list(subject_dict.keys()):
                    print(f'removing bad subject: {bad_subj}')
                    del subject_dict[bad_subj]
        
        if self.hparams.limit_training_samples:
            train_names = np.random.choice(train_names, size=self.hparams.limit_training_samples, replace=False, p=None)
        
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
        
        self.train_dataset = Dataset(**params,subject_dict=train_dict,use_augmentations=False, train=True)
        # load train mean/std of target labels to val/test dataloader
        self.val_dataset = Dataset(**params,subject_dict=val_dict,use_augmentations=False,train=False) 
        self.test_dataset = Dataset(**params,subject_dict=test_dict,use_augmentations=False,train=False)

        print("number of train_subj:", len(train_dict))
        print("number of test_subj:", len(test_dict))
        print("length of train_idx:", len(self.train_dataset.data))
        print("length of test_idx:", len(self.test_dataset.data))

        # 打印类别分布
        def print_class_distribution(dataset, name):
            distribution = defaultdict(int)
            # 解包元组的最后一个元素作为标签
            for item in dataset.data:
                # 假设标签是元组的最后一个元素
                label = item[-1]
                distribution[label] += 1
            print(f"{name} class distribution:", {k: v for k, v in sorted(distribution.items())})

        print_class_distribution(self.train_dataset, "Train")
        print_class_distribution(self.val_dataset, "Validation")
        print_class_distribution(self.test_dataset, "Test")
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": False if self.hparams.dataset_name == 'Dummy' else (train and (self.hparams.strategy == 'ddp')),
                "shuffle": train
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
        

    def train_dataloader(self):
        return self.train_loader

    # def val_dataloader(self):
    #     # return self.val_loader
    #     # currently returns validation and test set to track them during training
    #     return [self.val_loader, self.test_loader]
    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        return self.val_loader
    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        # parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        # group = parser.add_argument_group("DataModule arguments")
        # 创建一个参数组而不是新的解析器
        group = parent_parser.add_argument_group("fMRIDataModule")
        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default=r"E:\Python\File\Combined_fmri_dti_pretrained\Data", help="path to image datasets preprocessed for SwiFT")#改输入文件
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--input_type", default="rest",choices=['rest','task'],help='refer to datasets.py')
        group.add_argument("--train_split", default=0.8, type=float)
        group.add_argument("--val_split", default=0.1, type=float)
        group.add_argument("--batch_size", type=int, default=4)
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[80, 80, 80, 20], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")#改扩展维度
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        group.add_argument("--data_seed", type=int, default=42,
                           help="随机种子用于数据划分")
        return parent_parser


class FA3DDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # 创建划分目录
        split_dir_path = f'./Splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(
            split_dir_path,
            f"split_fixed_{self.hparams.dataset_split_num}.txt"
        )
        self.setup()

    def get_dataset(self):
        """返回对应的3D数据集类"""
        if self.hparams.dataset_name == "FA3D":
            return FA3DDataset
        else:
            raise NotImplementedError(f"Dataset {self.hparams.dataset_name} not implemented")

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        """将主题名称转换为索引列表"""
        subj_idx = np.array([str(x[0]) for x in subj_list])
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx

    def save_split(self, sets_dict):
        """保存数据集划分到文件"""
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def determine_split_randomly(self, S):
        """划分训练集和测试集，验证集设为空"""
        subjects = list(S.keys())
        labels = [S[subject][1] for subject in subjects]

        # 单次分层划分：训练集 vs 测试集
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - self.hparams.train_split,
            random_state=self.hparams.data_seed
        )

        train_idx, test_idx = next(sss.split(subjects, labels))
        S_train = [subjects[i] for i in train_idx]
        S_test = [subjects[i] for i in test_idx]
        S_val = []  # 验证集设为空列表

        # 保存划分结果（验证集为空）
        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test

    def load_split(self):
        """从文件加载划分"""
        if not os.path.exists(self.split_file_path):
            return None, None, None

        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x.strip() for x in subject_order]

        sections = {}
        current_section = None
        for line in subject_order:
            if line.endswith('_subjects'):
                current_section = line
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line)

        return (
            sections.get("train_subjects", []),
            sections.get("val_subjects", []),
            sections.get("test_subjects", [])
        )

    def make_subject_dict(self):
        """直接从目录结构创建主题字典，不依赖外部CSV文件"""
        final_dict = {}

        # 定义数据集名称到标签的映射
        dataset_labels = {
            "CN": 0,
            "PD": 1,
            "Prodromal": 2
        }

        # 打印调试信息
        print(f"图像根目录: {self.hparams.image_path}")

        # 遍历三个数据集目录
        subject_count = 0
        for dataset_name, label in dataset_labels.items():
            # 修改为使用预处理后的目录
            dataset_path = os.path.join(self.hparams.image_path, f"{dataset_name}_FA_processed", "img")
            print(f"检查预处理数据集目录: {dataset_path}")

            if not os.path.exists(dataset_path):
                print(f"警告: 预处理数据集路径不存在 - {dataset_path}")
                continue

            # 获取所有受试者目录
            subjects = [d for d in os.listdir(dataset_path)
                        if os.path.isdir(os.path.join(dataset_path, d))]

            if not subjects:
                print(f"警告: 预处理数据集目录下没有子目录 - {dataset_path}")
                # 列出目录内容以帮助诊断
                dir_contents = os.listdir(dataset_path)
                print(f"目录内容: {dir_contents[:10]}... (共{len(dir_contents)}项)")
                continue

            print(f"在 {dataset_name} 中找到 {len(subjects)} 个受试者")

            for subject in subjects:
                # 使用数据集前缀确保受试者名称唯一
                unique_subject = f"{dataset_name}_{subject}"

                # 标签直接使用数据集定义的标签
                final_dict[unique_subject] = [0, label]
                subject_count += 1

        # 打印主题统计
        print(f"总共找到 {subject_count} 个受试者")

        # 打印每个类别的样本数
        from collections import Counter
        label_counts = Counter([v[1] for v in final_dict.values()])
        print(f"类别分布: {dict(label_counts)}")

        if subject_count == 0:
            print("错误: 没有找到任何受试者！请检查")
            # 列出图像根目录下的内容
            root_contents = os.listdir(self.hparams.image_path)
            print(f"图像根目录内容: {root_contents[:10]}... (共{len(root_contents)}项)")
        return final_dict

    def prepare_data(self):
        """全局数据处理（如下载）"""
        pass

    def setup(self, stage=None):
        """设置数据集"""
        Dataset = self.get_dataset()

        # 创建主题字典
        subject_dict = self.make_subject_dict()

        # 加载或创建划分
        train_names, val_names, test_names = self.load_split()
        if train_names is None:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)

        # 筛选有效主题
        train_dict = {k: v for k, v in subject_dict.items() if k in train_names}
        val_dict = {k: v for k, v in subject_dict.items() if k in val_names}
        test_dict = {k: v for k, v in subject_dict.items() if k in test_names}

        params = {
            "root": self.hparams.image_path,
            "sequence_length": self.hparams.sequence_length,
            "contrastive": self.hparams.use_contrastive,
            "contrastive_type": self.hparams.contrastive_type,
            "stride_between_seq": self.hparams.stride_between_seq,
            "stride_within_seq": self.hparams.stride_within_seq,
            "with_voxel_norm": self.hparams.with_voxel_norm,
            "downstream_task": self.hparams.downstream_task,
            "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
            "input_type": self.hparams.input_type,
            "label_scaling_method": self.hparams.label_scaling_method,
            "dtype": 'float16'}

        self.train_dataset = Dataset(**params, subject_dict=train_dict, use_augmentations=False, train=True)
        # load train mean/std of target labels to val/test dataloader
        self.val_dataset = Dataset(**params, subject_dict=val_dict, use_augmentations=False, train=False)
        self.test_dataset = Dataset(**params, subject_dict=test_dict, use_augmentations=False, train=False)

        # 打印数据集统计信息
        print(f"Train subjects: {len(train_dict)}")
        print(f"Validation subjects: {len(val_dict)}")
        print(f"Test subjects: {len(test_dict)}")

        # 打印类别分布
        def print_distribution(dataset, name):
            counts = defaultdict(int)
            for _, label in dataset.subject_dict.values():
                counts[label] += 1
            print(f"{name} class distribution: {dict(counts)}")

        print_distribution(self.train_dataset, "Train")
        print_distribution(self.val_dataset, "Validation")
        print_distribution(self.test_dataset, "Test")

        # 配置数据加载器参数
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "pin_memory": True,
                "persistent_workers": True,
                "shuffle": train
            }

        self.train_loader = DataLoader(self.train_dataset, **get_params(True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(False))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader

    @classmethod
    def add_data_specific_args(cls, parent_parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        # group = parser.add_argument_group("DataModule arguments")
        # 创建一个参数组而不是新的解析器
        group = parent_parser.add_argument_group("FA3DDataModule")

        group.add_argument("--dataset_split_num", type=int, default=1) # dataset split, choose from 1, 2, or 3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax","standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default=r"E:\Python\File\Combined_fmri_dti_pretrained\Data", help="path to image datasets preprocessed for SwiFT")#改输入文件
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--input_type", default="rest",choices=['rest','task'],help='refer to datasets.py')
        group.add_argument("--train_split", default=0.8, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=4)#4
        group.add_argument("--eval_batch_size", type=int, default=16)
        group.add_argument("--img_size", nargs="+", default=[80, 80, 80], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")#改扩展维度
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=8)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        group.add_argument("--data_seed", type=int, default=42,
                           help="随机种子用于数据划分")
        return parent_parser


class MultimodalDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.pretraining:
            split_dir_path = f'./Splits/{self.hparams.dataset_name}/pretraining'
        else:
            split_dir_path = f'./Splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")

        self.setup()

    def get_dataset(self):
        return MultimodalDataset

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        subj_idx = np.array([str(x[1]) for x in subj_list])
        S = np.unique([x[1] for x in subj_list])
        print('unique subjects:', len(S))
        train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
        val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
        test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()
        return train_idx, val_idx, test_idx

    def save_split(self, sets_dict):
        with open(self.split_file_path, "w+") as f:
            for name, subj_list in sets_dict.items():
                f.write(name + "\n")
                for subj_name in subj_list:
                    f.write(str(subj_name) + "\n")

    def determine_split_randomly(self, S):
        subjects = list(S.keys())
        labels = [S[subject][1] for subject in subjects]
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - self.hparams.train_split,
            random_state=self.hparams.data_seed
        )

        train_idx, test_idx = next(sss.split(subjects, labels))
        S_train = [subjects[i] for i in train_idx]
        S_test = [subjects[i] for i in test_idx]
        S_val = []

        self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
        return S_train, S_val, S_test

    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1: val_index]
        val_names = subject_order[val_index + 1: test_index]
        test_names = subject_order[test_index + 1:]
        return train_names, val_names, test_names

    def prepare_data(self):
        return

    def make_subject_dict(self):
        final_dict = dict()

        dataset_labels = {
            "CN": 0,
            "PD": 1,
            "Prodromal": 2
        }

        for dataset_name, label in dataset_labels.items():
            fmri_path = os.path.join(self.hparams.image_path, f"{dataset_name}_MNI_to_TRs_minmax")
            fa_path = os.path.join(self.hparams.image_path, f"{dataset_name}_FA_processed", "img")

            # 检查路径是否存在
            if not os.path.exists(fmri_path):
                print(f"警告: fMRI路径不存在 - {fmri_path}")
                continue
            if not os.path.exists(fa_path):
                print(f"警告: FA路径不存在 - {fa_path}")
                continue

            fmri_subjects = set()
            try:
                fmri_subjects = set([d for d in os.listdir(fmri_path)
                                     if os.path.isdir(os.path.join(fmri_path, d))])
            except Exception as e:
                print(f"读取fMRI受试者错误: {fmri_path}, {str(e)}")

            fa_subjects = set()
            try:
                fa_subjects = set([d for d in os.listdir(fa_path)
                                   if os.path.isdir(os.path.join(fa_path, d))])
            except Exception as e:
                print(f"读取FA受试者错误: {fa_path}, {str(e)}")

            common_subjects = fmri_subjects & fa_subjects

            for subject in common_subjects:
                unique_subject = f"{dataset_name}_{subject}"
                final_dict[unique_subject] = [0, label]

        return final_dict

    def setup(self, stage=None):
        Dataset = self.get_dataset()
        params = {
            "fmri_root": self.hparams.image_path,
            "fa_root": self.hparams.image_path,
            "sequence_length": self.hparams.sequence_length,
            "contrastive": self.hparams.use_contrastive,
            "contrastive_type": self.hparams.contrastive_type,
            "stride_between_seq": self.hparams.stride_between_seq,
            "stride_within_seq": self.hparams.stride_within_seq,
            "with_voxel_norm": self.hparams.with_voxel_norm,
            "downstream_task": self.hparams.downstream_task,
            "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
            "input_type": self.hparams.input_type,
            "label_scaling_method": self.hparams.label_scaling_method,
            "dtype": 'float16'
        }

        subject_dict = self.make_subject_dict()
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)

        if self.hparams.limit_training_samples and len(train_names) > self.hparams.limit_training_samples:
            train_names = np.random.choice(
                train_names,
                size=self.hparams.limit_training_samples,
                replace=False,
                p=None
            )

        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}

        self.train_dataset = Dataset(**params, subject_dict=train_dict, train=True)
        self.val_dataset = Dataset(**params, subject_dict=val_dict, train=False)
        self.test_dataset = Dataset(**params, subject_dict=test_dict, train=False)

        print("\n===== 数据集统计信息 =====")
        print(f"训练集: {len(self.train_dataset)} 个序列")
        print(f"验证集: {len(self.val_dataset)} 个序列")
        print(f"测试集: {len(self.test_dataset)} 个序列")
        print("=========================\n")

        # 数据加载器配置
        def get_params(train):
            shuffle = train  # 训练时启用全局shuffle
            persistent_workers = False if self.hparams.dataset_name == 'Dummy' else (
                    train and (self.hparams.strategy == 'ddp'))

            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": persistent_workers,
                "shuffle": shuffle
            }

        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_loader

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        group = parent_parser.add_argument_group("MultimodalDataModule")
        group.add_argument("--dataset_split_num", type=int, default=1)  # 数据集分割编号，选择1,2或3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax", "standardization"],
                           help="标签缩放方法")
        group.add_argument("--image_path", default=r"E:\Python\File\Combined_fmri_dti_pretrained\Data",
                           help="图像数据根目录")
        group.add_argument("--input_type", default="rest", choices=['rest', 'task'],
                           help='输入类型：静息态或任务态')
        group.add_argument("--train_split", default=0.8, type=float,
                           help="训练集比例")
        group.add_argument("--val_split", default=0.1, type=float,
                           help="验证集比例")
        group.add_argument("--batch_size", type=int, default=4,
                           help="训练批大小")
        group.add_argument("--eval_batch_size", type=int, default=16,
                           help="评估批大小")
        group.add_argument("--img_size", nargs="+", default=[80, 80, 80, 20], type=int,
                           help="图像尺寸 (D, H, W, T)")
        group.add_argument("--sequence_length", type=int, default=20,
                           help="fMRI序列长度")
        group.add_argument("--stride_between_seq", type=float, default=0.5,
                           help="序列间步长比例 (0.0-1.0)")
        group.add_argument("--stride_within_seq", type=int, default=1,
                           help="序列内步长 (连续帧为1)")
        group.add_argument("--num_workers", type=int, default=8,
                           help="数据加载工作线程数")
        group.add_argument("--with_voxel_norm", type=str2bool, default=False,
                           help="是否使用体素归一化")
        group.add_argument("--shuffle_time_sequence", action='store_true',
                           help="是否打乱时间序列")
        group.add_argument("--limit_training_samples", type=int, default=None,
                           help="限制训练受试者数量")
        group.add_argument("--data_seed", type=int, default=42,
                           help="用于数据划分的随机种子")
        return parent_parser