import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader
import argparse
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from DataModule.CIFAR10ImbalanceDataModule import CIFAR10ImbalanceDataModule

class CIFAR10Classifier(LightningModule):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU()
        )

        self.classifer = nn.Linear(in_features=128, out_features=10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.model(x)
        output = self.classifer(logits)
        return output, logits

    def training_step(self, batch, batch_idx):
        input, labels = batch
        output, _ = self(input)
        loss = self.criterion(output, labels)
        self.log("train_loss", loss)
        print(f"\ntrain_loss is {loss}")
        return loss

    def test_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        pred, _ = self(input)
        loss = self.criterion(pred, labels)
        acc = (pred.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss)
        print(f"\nval_loss is {loss}")
        self.log("acc", acc)
        print(f"\nacc is {acc}")

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=1e-4)


class TwoStageCIFAR10Classifier(LightningModule):
    def __init__(self, _args, _sim_geometric, _sorted_eigenvalues_list, _sorted_eigenvectors_list, tail_classes):
        super(TwoStageCIFAR10Classifier, self).__init__()
        self.args = _args
        self.sim_geometric = _sim_geometric # [] 10个 float
        # print(f"self.sim_geometric : {self.sim_geometric}")
        self.sorted_eigenvalues_list = _sorted_eigenvalues_list # [10个] array
        # print(f"self.sorted_eigenvalues_list: {len(self.sorted_eigenvalues_list)}")
        self.sorted_eigenvectors_list = _sorted_eigenvectors_list # [10个] array
        # print(f"self.sorted_eigenvectors_list: {len(self.sorted_eigenvectors_list)}")
        self.tail_classes = tail_classes
        # print(f"self.tail_classes; {self.tail_classes}")

        self.feature_extractor = CIFAR10Classifier().model
        self.classifer = CIFAR10Classifier().classifer

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        # 前馈过程
        # logits = self.feature_extractor(x)
        print(f"x shape : {x.shape}")
        output = self.classifer(x)
        return output

    def freeze_diff_stage(self, stage=0):
        if stage == 0:
            # 第二阶段 冻结feature_extractor
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.classifer.parameters():
                param.requires_grad = True

        if stage == 1:
            # 第三阶段 冻结classifer
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
            for param in self.classifer.parameters():
                param.requires_grad = False

    # 不确定性表示函数
    def feature_uncertainty_representation(self, x, c):
        """
           根据特征向量和特征值，对给定特征进行不确定性表示 获得增强之后的特征表示
           输入：x 之前的特征表征 c类别 label
           x [batch_size(15), 128] nt个尾部数据 nt(1+na)个头部数据 nt 3 na 2
           每一个尾部数据跟着na个头部数据
           尾部 0 1 2
           index: class c
        """
        nt = self.args.nt
        perturbed_features = []
        for i in range(nt):
            tail_feature = x[i]
            epsilon = np.random.normal(0, 1, self.sorted_eigenvalues_list[c[i]].shape)
            perturbed_value = np.dot(self.sorted_eigenvectors_list[c[i]], epsilon * self.sorted_eigenvalues_list[c[i]])
            perturbed_value_tensor = tail_feature + torch.tensor(perturbed_value).cuda()
            perturbed_features.append(perturbed_value_tensor.unsqueeze(0))
        return torch.cat([torch.cat(perturbed_features, dim=0), x[nt:]],dim=0)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        with torch.no_grad():
            original_logits = self.feature_extractor(images) # [bs(15), 128] nt个尾部 nt(1+na)个为头部数据
        # 确定训练阶段
        self.freeze_diff_stage()
        perturbed_features = self.feature_uncertainty_representation(original_logits, label)
        perturbed_features = perturbed_features.to(torch.float32)
        output = self(perturbed_features)
        loss = self.criterion(output, labels)
        self.log("train_loss", loss)
        print(f"\ntrain_loss is {loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        with torch.no_grad():
            original_logits = self.feature_extractor(input) # [bs(15), 128] nt个尾部 nt(1+na)个为头部数据
        pred = self(original_logits)
        loss = self.criterion(pred, labels)
        acc = (pred.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss)
        print(f"\nval_loss is {loss}")
        self.log("acc", acc)
        print(f"\nacc is {acc}")

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=1e-4)



def train(args):
    # 训练模型
    model = CIFAR10Classifier()

    print(f"torch_gpu is aviliable : {torch.cuda.is_available()}")

    # todo: 将训练好的模型保存
    # 创建 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        monitor='acc',  # 监控的指标
        dirpath=r'./checkpoint',  # 保存路径
        filename='best-checkpoint',  # 文件名模板
        save_top_k=1,  # 保存验证集最优的 k 个模型
        mode='max'  # 当 'val_loss' 越小越好时为 'min', 否则为 'max'
    )

    trainer = Trainer(max_epochs=3, callbacks=[checkpoint_callback])
    # trainer = Trainer(max_epochs=20, accelerator='gpu')
    trainer.fit(model, cifar10_train_dataloader, cifar10_test_dataloader)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="cifar10")
    parse.add_argument("--data_dir", default='./data', type=str, help="data_cifar 10 dir")
    parse.add_argument("--batch_size", default=24, type=int, help="batch size")
    parse.add_argument("--train", default=False, type=bool, help="whether is train process")
    parse.add_argument("--checkpoint", default='./checkpoint', type=str, help="checkpoint path")
    parse.add_argument("--num_classes", default=10, type=int, help="number of classes")
    parse.add_argument("--nt", default=3, type=int, help="nt for tail_num")
    parse.add_argument("--na", default=2, type=int, help="na for each tail class")

    args = parse.parse_args()
    # 尾部类
    tail_classes = [0, 1, 2, 3, 4]
    # 构建batch_size
    args.batch_size = args.nt + args.nt * (1 + args.na)
    print(f"batch_size: {args.batch_size}")

    cifar_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    cifar10_train_datasets = torchvision.datasets.CIFAR10(root=r'./data', train=True,
                                                          transform=cifar_transforms, download=True)
    cifar10_test_datasets = torchvision.datasets.CIFAR10(root=r'./data', train=False,
                                                         transform=cifar_transforms, download=True)

    cifar10_train_dataloader = DataLoader(dataset=cifar10_train_datasets, shuffle=True,
                                           batch_size=args.batch_size)
    cifar10_test_dataloader = DataLoader(dataset=cifar10_test_datasets, shuffle=True,
                                          batch_size=args.batch_size)

    if args.train is True:
        train(args)
    # 获取已经保存好的模型
    model = CIFAR10Classifier.load_from_checkpoint("./checkpoint/best-checkpoint.ckpt").cuda()
    # # 验证当前模型 计算与类别C最相似的类别Y
    model.eval()
    # # 保存每一个样本的pred
    pred_all = []
    label_all = []
    logits_all = []
    # # 验证
    with torch.no_grad():
        for image, label in cifar10_train_dataloader:
            image = image.cuda()
            label = label.cuda()
            pred, logits = model(image)
            pred_all.append(pred)
            label_all.append(label)
            logits_all.append(logits)

    # 将所有的logits合并为一个tensor
    pred_all_tensor = torch.cat(pred_all, dim=0) # shape [num, num_class]
    label_all_tensor = torch.cat(label_all, dim=0)
    logits_all_tensor = torch.cat(logits_all, dim=0)

    # 计算每一个类别的平均预测分数
    average_scores = torch.zeros(args.num_classes, args.num_classes)

    # 遍历每一个类别 计算该类别的平均预测分数
    for c in range(args.num_classes):
        class_mask = (label_all_tensor == c)
        class_logits = pred_all_tensor[class_mask] # 当前类别所有的logits
        # 计算所有logits的平均值
        average_avg_scores = class_logits.mean(dim=0)
        average_scores[c] = average_avg_scores

    # 寻找最相似的类别（除了当前类别之外，平均logit分数最大的别的非当前头（尾部）类别）
    most_sim_classes = []
    for c in range(args.num_classes):
        avg_scores_c = average_scores[c]
        avg_scores_c[c] = float('-inf')# 当前类别置为负无穷大，不参与比较
        if c in tail_classes:
            # c是尾部类，将所有尾部类设置为负无穷大 不参与比较
            for c_idx in range(args.num_classes):
                if c_idx in tail_classes:
                    avg_scores_c[c_idx] = float('-inf')
        else:
            # c是头部类，将所有的头部类都设置为负无穷大 不参与比较
            for c_idx in range(args.num_classes):
                if c_idx not in tail_classes:
                    avg_scores_c[c_idx] = float('-inf')
        most_similar_class = torch.argmax(avg_scores_c).item()# 最相似类别
        most_sim_classes.append(most_similar_class)
        print(f"Class {c} is most similar to class {most_similar_class}")

    print("Most similar classes for each class:", most_sim_classes)

    # 相似类的logits集合 后续利用它们指导尾部类的分布恢复 如果是头部类 则0 是头部类 1 是尾部类；
    # 如果是尾部类 则0 是尾部类 1是头部类
    sim_logits_matrix = [] # [[shape(m, 128), shape(n, 128)],...]
    for c in range(args.num_classes):
        class_mask = (label_all_tensor == c)
        class_sim_mask = (label_all_tensor == most_sim_classes[c])
        src_class_logits = logits_all_tensor[class_mask]
        tar_class_logits = logits_all_tensor[class_sim_mask]
        print(f"\nclass {c} is src_class_logits : {src_class_logits.shape}, tar_class_logits : {tar_class_logits.shape}")
        sim_logits_matrix.append([src_class_logits, tar_class_logits])

    # 计算他们的协方差矩阵
    sim_geometric = [] # 存储相似度
    sorted_eigenvalues_list = [] # 类别C的特征值[[values], [values],...]
    sorted_eigenvectors_list = [] # 类别c的特征向量 [[128, 128], [128, 128], ...]
    for c in range(args.num_classes):
        src_class_logits = sim_logits_matrix[c][0]
        tar_class_logits = sim_logits_matrix[c][1]

        # 计算协方差矩阵
        src_covariance_matrix = np.cov(src_class_logits.cpu(), rowvar=False)
        tar_covariance_matrix = np.cov(tar_class_logits.cpu(), rowvar=False)

        print(f"\nclass {c} is src_covariance_matrix : {src_covariance_matrix.shape}, tar_covariance_matrix : {tar_covariance_matrix.shape}")
        # 进行特征值分解
        src_eigenvalues, src_eigenvectors = np.linalg.eigh(src_covariance_matrix)
        tar_eigenvalues, tar_eigenvectors = np.linalg.eigh(tar_covariance_matrix)
        # print(f"\n src_eigenvalues = {src_eigenvalues}, src_eigenvectors = {src_eigenvectors.shape}")
        # print(f"\n tar_eigenvalues = {tar_eigenvalues}, tar_eigenvectors = {tar_eigenvectors.shape}")

        # 对特征值进行排序
        src_sorted_indices = np.argsort(src_eigenvalues)[::-1]
        src_eigenvalues = src_eigenvalues[src_sorted_indices]
        src_eigenvectors = src_eigenvectors[:, src_sorted_indices]

        src_sorted_indices = np.argsort(tar_eigenvalues)[::-1]
        tar_eigenvalues = tar_eigenvalues[src_sorted_indices]
        tar_eigenvectors = tar_eigenvectors[:, src_sorted_indices]
        # 存储排序以后的特征值和特征向量 主要面向 尾部类 -》头部类
        sorted_eigenvalues_list.append(tar_eigenvalues)
        sorted_eigenvectors_list.append(tar_eigenvectors)

        print("sorted")
        similarity = 0
        for i in range(len(tar_eigenvectors)):
            similarity += np.abs(np.dot(src_eigenvectors[:, i].T, tar_eigenvectors[:, i]))
        sim_geometric.append(similarity)
        print("Similarity of the geometric shapes of the two perceptual manifolds:", similarity)

    print(f"similarity of geometric list {sim_geometric}")


    ###################第二阶段 重塑决策边界############################
    finetuning_model = TwoStageCIFAR10Classifier(_args=args, _sim_geometric=sim_geometric,
                                                 _sorted_eigenvalues_list=sorted_eigenvalues_list,
                                                 _sorted_eigenvectors_list=sorted_eigenvectors_list,
                                                 tail_classes=tail_classes)

    cifar10_lt_dataset = CIFAR10ImbalanceDataModule(tail_classes=tail_classes)
    cifar10_lt_dataset.prepare_data()
    cifar10_lt_dataset.setup()

    # 创建 ModelCheckpoint 回调
    checkpoint_callback2 = ModelCheckpoint(
        monitor='acc',  # 监控的指标
        dirpath=r'./tuning_checkpoint',  # 保存路径
        filename='best-checkpoint',  # 文件名模板
        save_top_k=1,  # 保存验证集最优的 k 个模型
        mode='max'  # 当 'val_loss' 越小越好时为 'min', 否则为 'max'
    )

    # trainer = Trainer(max_epochs=3)
    trainer = Trainer(max_epochs=20, accelerator='gpu', callbacks=[checkpoint_callback2])
    trainer.fit(finetuning_model, cifar10_lt_dataset)
