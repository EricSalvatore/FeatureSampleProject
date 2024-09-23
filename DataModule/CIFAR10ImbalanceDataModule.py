import torch
from pytorch_lightning import LightningDataModule
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, Subset, DataLoader, Sampler
from torchvision import datasets, transforms

# 自定义采样器
class CIFAR10Sampler(Sampler):
    """
    自定义采样器，每一次从数据中取nt个尾部数据 nt(1+na)个头部数据
    """
    def __init__(self, tail_indices, head_indices, nt, na):
        self.tail_indices = tail_indices
        self.head_indices = head_indices
        self.nt = nt
        self.na = na
        self.num_batches = len(self.tail_indices) // self.nt  # 批次数

    def __iter__(self):
        np.random.shuffle(self.tail_indices)
        np.random.shuffle(self.head_indices)

        for i in range(self.num_batches):
            # 从尾部采样nt个数据
            tail_batch = self.tail_indices[i * self.nt:(i + 1) * self.nt]

            # 从头部数据采样nt(1+na)个数据
            head_start_idx = i * self.nt * (1 + self.na)
            head_end_idx = (i + 1) * self.nt * (1 + self.na)
            head_batch = self.head_indices[head_start_idx:head_end_idx]

            # 组合为一个batch
            yield list(tail_batch) + list(head_batch)

    def __len__(self):
        return self.num_batches


# 设定类别0 1 2 3 4为尾部类
# 构建CIFAR10的尾部类数据
class CIFAR10ImbalanceDataModule(LightningDataModule):
    def __init__(self, batch_size, tail_classes, tail_ratio=0.1, data_dir='./data', nt=3, na=3):
        super().__init__()
        self.data_dir = data_dir
        self.tail_classes = tail_classes # [0, 1, 2, 3, 4]
        self.tail_ratio = tail_ratio
        self.batch_size = batch_size
        self.nt = nt
        self.na = na

        # 数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # 数据集
        self.train_datasets = None
        self.test_datasets = None

    def prepare_data(self):
        # 准备数据
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # 加载训练集 和 验证集
        full_train_dataset = datasets.CIFAR10(root=self.data_dir, train=True, transform=self.transform)
        val_dataset = datasets.CIFAR10(root=self.data_dir, train=False, transform=self.transform)

        # 根据给定的尾部类和头部类构建不平衡的训练集
        self.tail_class_indices = []  # 保存尾部类样本的索引
        self.head_class_indices = []  # 保存头部类样本的索引

        for tail_class in self.tail_classes:
            # 获取当前尾部类的所有样本索引
            class_indices = np.where(np.array(full_train_dataset.targets) == tail_class)[0]
            # 按比例随机采样该类别的样本索引
            num_samples = int(len(class_indices) * self.tail_ratio)
            sample_indices = np.random.choice(class_indices, num_samples, replace=False)
            self.tail_class_indices.extend(sample_indices)

        for head_class in range(10):
            if head_class not in self.tail_classes:
                # 获取当前头部类的所有样本索引
                class_indices = np.where(np.array(full_train_dataset.targets) == head_class)[0]
                self.head_class_indices.extend(class_indices)

        # 组合尾部类和头部类的索引，创建不平衡数据集
        imbalanced_indices = np.array(self.tail_class_indices + self.head_class_indices)
        np.random.shuffle(imbalanced_indices)  # 打乱样本索引

        # 创建不平衡数据集
        self.train_datasets = Subset(full_train_dataset, imbalanced_indices)
        self.test_datasets = val_dataset

    def train_dataloader(self):
        sampler = CIFAR10Sampler(self.tail_class_indices, self.head_class_indices, self.nt, self.na)
        return DataLoader(self.train_datasets, batch_size=self.nt + self.nt * (1 + self.na), sampler=sampler)

    def val_dataloader(self):
        return DataLoader(self.test_datasets, batch_size=self.nt + self.nt * (1 + self.na), shuffle=False)






