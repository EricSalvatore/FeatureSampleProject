import sys
from os import system
import os
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
import torch.nn as nn
from torchmetrics import MeanMetric
from torchvision import transforms
from torchvision import models
import torch.multiprocessing as mp

from torch.utils.data import DataLoader
import argparse
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import clip
from torchvision.datasets import CocoDetection
import cv2
import torch.nn.functional as F
from pycocotools import mask as maskUtils

import DurModule
from DataModule.CIFAR10ImbalanceDataModule import CIFAR10ImbalanceDataModule
# PyTorch Lightning 模型
class SegmentationLightningModel(pl.LightningModule):
    def __init__(self, num_classes, _model_features, _device):
        super(SegmentationLightningModel, self).__init__()
        self.backbone = _model_features.to(_device)
        self._device = _device

        self.pro1 = nn.Linear(512, 12544)
        self.seg_head = nn.Conv2d(1, num_classes, kernel_size=1)
        self.avg_acc = MeanMetric()

    def forward(self, x):
        logits = self.backbone(x)
        pro_logits = self.pro1(logits)
        pro_logits = pro_logits.unsqueeze(1)
        pro_logits = pro_logits.view(pro_logits.size(0), 1, 112, 112)
        pred =self.seg_head(pro_logits)
        return logits, pred

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images, targets = images.to(self._device), targets.to(self._device)
        _, outputs = self(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        print(f"train_loss is {loss:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images, targets = images.to(self._device), targets.to(self._device)
        _, outputs = self(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)
        print(f"val_loss is {loss:.4f}")

        pred_masks = torch.argmax(outputs, dim=1)  # 获取类别预测
        targets = targets.squeeze(1)  # 去掉通道维度以匹配形状

        # 计算 IoU
        intersection = (pred_masks == targets).float().sum((1, 2))  # 每个样本的交集
        union = ((pred_masks != 0) | (targets != 0)).float().sum((1, 2))

        # 避免除以零
        iou = intersection / (union + 1e-6)  # 加上小值避免除以零

        mean_iou = iou.mean()  # 计算平均 IoU

        # 记录损失和 IoU
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)
        self.log('mean_iou', mean_iou)
        self.avg_acc(mean_iou)
        print('mean_iou: ', mean_iou)
        print('val_loss: ', loss)


    def on_validation_epoch_end(self):
        avg_acc = self.avg_acc.compute()
        print(f"avg acc is {avg_acc}")
        self.log("avg_acc", avg_acc)
        self.avg_acc.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

class TwoStageCIFAR10Classifier(LightningModule):
    def __init__(self, _args, _model, _sim_geometric, _sorted_eigenvalues_list, _sorted_eigenvectors_list, tail_classes):
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

        self.backbone = _model.backbone
        self.seg_head = _model.seg_head

        self.criterion = nn.CrossEntropyLoss()
        self.avg_acc = MeanMetric()

    def forward(self, x):
        # 前馈过程
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.seg_head(x)
        return output

    def freeze_diff_stage(self, stage=0):
        if stage == 0:
            # 第二阶段 冻结backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.seg_head.parameters():
                param.requires_grad = True

        if stage == 1:
            # 第三阶段 冻结seg_head
            for param in self.backbone.parameters():
                param.requires_grad = True
            for param in self.seg_head.parameters():
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
            original_logits = self.backbone(images) # [bs(15), 128] nt个尾部 nt(1+na)个为头部数据
        # 确定训练阶段
        self.freeze_diff_stage()
        original_logits = original_logits.view(images.size(0), -1)
        perturbed_features = self.feature_uncertainty_representation(original_logits, label)
        perturbed_features = perturbed_features.to(torch.float32)
        output = self(perturbed_features)
        loss = self.criterion(output, labels)
        self.log("train_loss", loss)
        print(f"train_loss is {loss:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        input, labels = batch
        with torch.no_grad():
            original_logits = self.backbone(input) # [bs(15), 128] nt个尾部 nt(1+na)个为头部数据
        original_logits = original_logits.view(input.size(0), -1)
        pred = self(original_logits)
        loss = self.criterion(pred, labels)
        acc = (pred.argmax(dim=1) == labels).float().mean()

        self.log("val_loss", loss)
        print(f"val_loss is {loss:.4f}")
        self.log("acc", acc)
        print(f"acc is {acc:.4f}")
        self.avg_acc(acc)

    def on_validation_epoch_end(self):
        avg_acc_value = self.avg_acc.compute()
        print(f"avg_acc_value is {avg_acc_value}")
        self.log("avg_acc", avg_acc_value)
        self.avg_acc.reset()

    def configure_optimizers(self):
        return Adam(params=self.parameters(), lr=1e-4)

def train(args, _device, _train_loader, _val_loader):
    # 训练模型
    # 使用clip
    pretrained_model, process = clip.load(args.clip_image_name, device=_device)

    pretrained_vit_model = pretrained_model.visual.float()
    model = SegmentationLightningModel(num_classes=args.num_classes, _model_features=pretrained_vit_model, _device=device)

    print(f"torch_gpu is aviliable : {torch.cuda.is_available()}")

    # todo: 将训练好的模型保存
    # 创建 ModelCheckpoint 回调
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_acc',  # 监控的指标
        dirpath=r'./clip_seg_checkpoint',  # 保存路径
        filename=f'best-{args.model_name}-checkpoint-epoch-{args.pretrained_epoch}-best-acc-'+'{avg_acc:.3f}',  # 文件名模板
        save_top_k=1,  # 保存验证集最优的 k 个模型
        mode='max'  # 当 'val_loss' 越小越好时为 'min', 否则为 'max'
    )

    trainer = Trainer(max_epochs=args.pretrained_epoch, devices=[4], accelerator='gpu', callbacks=[checkpoint_callback])
    # todo:长尾数据开关
    trainer.fit(model, _train_loader, _val_loader)

def create_segmentation_mask(target, image_size=(224, 224), _devices = 'cuda'):
    mask = torch.zeros(image_size, dtype=torch.uint8, device=_devices)

    for obj in target:
        if 'segmentation' in obj:
            segmentation = obj['segmentation']
            if isinstance(segmentation, list):  # 多边形格式
                for seg in segmentation:
                    polygon = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    polygon_tensor = torch.tensor(polygon, device=_devices)
                    cv2.fillPoly(mask.cpu().numpy(), polygon_tensor.cpu().numpy(), color=obj['category_id'])  # 填充颜色  # 填充颜色
            elif isinstance(segmentation, dict) and 'counts' in segmentation:  # RLE格式
                rle = maskUtils.frPyObjects(segmentation, image_size[0], image_size[1])
                rle_mask = maskUtils.decode(rle)

                # 调整 RLE 掩码的大小
                rle_mask_resized = cv2.resize(rle_mask, image_size[::-1], interpolation=cv2.INTER_NEAREST)
                mask = torch.maximum(mask, torch.tensor(rle_mask_resized, device=_devices))  # 结合 RLE 掩码
        else:
            print("No segmentation data found for this object.")

    return mask.long()


def collate_fn(batch):
    images, targets = zip(*batch)
    images = [img for img in images]
    target_masks = [create_segmentation_mask(t) for t in targets]
    return torch.stack(images).to('cuda'), torch.stack(target_masks).to('cuda')

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="coco")
    parse.add_argument("--model_name", default='resnet18', type=str, help="model name")
    parse.add_argument("--clip_image_name", default='ViT-B/32', type=str, help="CLIP image model name")
    parse.add_argument("--data_dir", default='./data/coco1000/coco2017_1000', type=str, help="CLIP image model name")
    parse.add_argument("--train_ann_file", default='annotations/instances_train2017.json', type=str, help="train ann path")
    parse.add_argument("--val_ann_file", default='annotations/instances_val2017.json', type=str, help="val anno path")

    parse.add_argument("--batch_size", default=96, type=int, help="batch size")
    parse.add_argument("--pretrained_epoch", default=100, type=int, help="pretrained_epoch")
    parse.add_argument("--tuning_epoch", default=100, type=int, help="tuning epoch")
    parse.add_argument("--num_classes", default=91, type=int, help="num_classes")

    # 开关
    parse.add_argument("--train", default=True, type=bool, help="whether is train process")
    parse.add_argument("--is_lt", default=False, type=bool, help="whether is is_lt dataset")
    parse.add_argument("--end_train", default=True, type=bool, help="only pretrain")

    parse.add_argument("--nt", default=3, type=int, help="nt for tail_num")
    parse.add_argument("--na", default=2, type=int, help="na for each tail class")

    args = parse.parse_args()
    # 设置多进程的启动方法
    mp.set_start_method('spawn', force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(42)
    # 指定下载路径
    torch.hub.set_dir('./downloaded_model')  # 替换为你的下载路径
    if args.is_lt is True:
        # 构建batch_size
        args.batch_size = args.nt + args.nt * (1 + args.na)
        print(f"batch_size: {args.batch_size}")
    else:
        print(f"batch_size: {args.batch_size}")
    # 数据增强和预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    data_dir = args.data_dir
    train_ann_file = os.path.join(data_dir, args.train_ann_file)
    val_ann_file = os.path.join(data_dir, args.val_ann_file)
    # 加载数据集
    train_dataset = CocoDetection(root=os.path.join(data_dir, "train2017"),
                                  annFile=train_ann_file,
                                  transform=transform)

    val_dataset = CocoDetection(root=os.path.join(data_dir, "val2017"),
                                annFile=val_ann_file,
                                transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    if args.train is True:
        train(args, _device= device , _train_loader=train_loader, _val_loader=val_loader)

    if args.end_train is True:
        print(f"end file")
        sys.exit()
    # # 获取已经保存好的模型
    # # 获取模型
    # print(f"开始加载视觉大模型")
    # pretrained_model, process = clip.load(args.clip_image_name, device=device)
    # pretrained_vit_model = pretrained_model.visual.float()
    # print(f"视觉大模型加载完成")
    #
    # print(f"开始加载预训练分类模型")
    # model = CIFAR10Classifier.load_from_checkpoint(f"./clip_checkpoint/best-ViT-B/32-checkpoint-epoch-100-best-acc-avg_acc=0.787.ckpt",
    #                                                _model=pretrained_vit_model).cuda()
    # print(f"分类预训练模型加载完成")
    # # # 验证当前模型 计算与类别C最相似的类别Y
    # # model = CIFAR10Classifier()
    # model.eval()
    # # # 保存每一个样本的pred
    # pred_all = []
    # label_all = []
    # logits_all = []
    # # # 验证
    # print(f"开始整合预训练模型的推理结果")
    # with torch.no_grad():
    #     for image, label in cifar10_train_dataloader:
    #         image = image.cuda()
    #         label = label.cuda()
    #         pred, logits = model(image)
    #         pred_all.append(pred)
    #         label_all.append(label)
    #         logits_all.append(logits)
    #
    # # 将所有的logits合并为一个tensor
    # pred_all_tensor = torch.cat(pred_all, dim=0) # shape [num, num_class]
    # label_all_tensor = torch.cat(label_all, dim=0)
    # logits_all_tensor = torch.cat(logits_all, dim=0)
    # print(f"预训练模型推理结果整合完成")
    #
    # # 获取得到与之最相似的类别：示例：List  [8, 9, 5, 5, 7, 3, 3, 4, 0, 1]
    # # 如果index是尾部类 则找到最相似的头部类 List[0] = 8： 尾部类0的最相似的头部类为8
    # # 如果index是头部类 则找到最相似的尾部类 List[5] = 3: 头部类5的最相似mkkj的尾部类为3
    # print(f"开始获取相似类列表")
    # most_sim_classes = DurModule.search_most_similar_class(args=args, pred_all_tensor=pred_all_tensor,
    #                                     label_all_tensor=label_all_tensor,
    #                                     logits_all_tensor=logits_all_tensor)
    #
    # print(f"most_sim_classes: {most_sim_classes}")
    # print(f"相似类列表获取完成")
    #
    # # 根据最相似的类别，计算协方差矩阵 得到与目标类别对应的最相似类别的特征向量、特征值、特征相似度
    # print(f"计算相似类特征向量、特征值、特征相似度")
    # sorted_eigenvalues_list, sorted_eigenvectors_list, sim_geometric \
    #     = DurModule.get_eigenvalues_and_eigenvectors_and_sim_geometric(args=args,
    #                                                                    most_sim_classes=most_sim_classes,
    #                                                                    label_all_tensor=label_all_tensor,
    #                                                                    logits_all_tensor=logits_all_tensor)
    # print(f"相似类特征向量、特征值、特征相似度计算完成")
    # ###################第二阶段 重塑决策边界############################
    # print(f"开始第二阶段 重塑决策边界")
    # # 获取模型
    # print(f"开始初始化二阶段微调模型")
    # finetuning_model = TwoStageCIFAR10Classifier(_args=args, _model=model, _sim_geometric=sim_geometric,
    #                                              _sorted_eigenvalues_list=sorted_eigenvalues_list,
    #                                              _sorted_eigenvectors_list=sorted_eigenvectors_list,
    #                                              tail_classes=tail_classes)
    # print(f"初始化二阶段微调模型完成")
    #
    # if args.is_lt is True:
    #     cifar10_lt_dataset = CIFAR10ImbalanceDataModule(tail_classes=tail_classes)
    #     cifar10_lt_dataset.prepare_data()
    #     cifar10_lt_dataset.setup()
    #
    # # 创建 ModelCheckpoint 回调
    # checkpoint_callback2 = ModelCheckpoint(
    #     monitor='avg_acc',  # 监控的指标
    #     dirpath=r'./clip_tuning_checkpoint',  # 保存路径
    #     filename='best-clip-checkpoint-epoch-{epoch:03d}-acc-{avg_acc:.3f}',  # 文件名模板
    #     save_top_k=1,  # 保存验证集最优的 k 个模型
    #     mode='max'  # 当 'val_loss' 越小越好时为 'min', 否则为 'max'
    # )
    #
    # trainer = Trainer(max_epochs=args.tuning_epoch, accelerator='gpu', strategy='ddp_find_unused_parameters_true', devices=[0], callbacks=[checkpoint_callback2])
    # # todo: 长尾数据开关
    # print(f"微调模型训练开始")
    # if args.is_lt:
    #     trainer.fit(finetuning_model, cifar10_lt_dataset)
    # else:
    #     trainer.fit(finetuning_model, cifar10_train_dataloader, cifar10_test_dataloader)
    # print(f"微调模型训练结束")

