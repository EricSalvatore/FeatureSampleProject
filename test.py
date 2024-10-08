import os
import torch
import torch.nn as nn
from torchvision.models import resnet34
import pytorch_lightning as pl
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from DataModule import CoCoDatasetsModule
from DataModule.CoCoDatasetsModule import CocoDataModule
from pycocotools import mask as maskUtils
import numpy as np
import cv2
import torch.nn.functional as F

data_dir = "./data/coco1000/coco2017_1000"
train_ann_file = os.path.join(data_dir, "annotations/instances_train2017.json")
val_ann_file = os.path.join(data_dir, "annotations/instances_val2017.json")

# 数据增强和预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def create_segmentation_mask(target, image_size=(256, 256)):
    mask = np.zeros(image_size, dtype=np.uint8)

    for obj in target:
        if 'segmentation' in obj:
            segmentation = obj['segmentation']
            if isinstance(segmentation, list):  # 多边形格式
                for seg in segmentation:
                    polygon = np.array(seg, dtype=np.int32).reshape((-1, 1, 2))
                    # print("obj['category_id']: ", obj['category_id'])
                    cv2.fillPoly(mask, polygon, color=obj['category_id'])  # 填充颜色
            elif isinstance(segmentation, dict) and 'counts' in segmentation:  # RLE格式
                rle = maskUtils.frPyObjects(segmentation, image_size[0], image_size[1])
                rle_mask = maskUtils.decode(rle)

                # 调整 RLE 掩码的大小
                rle_mask_resized = cv2.resize(rle_mask, image_size[::-1], interpolation=cv2.INTER_NEAREST)
                mask = np.maximum(mask, rle_mask_resized)  # 结合 RLE 掩码
        else:
            print("No segmentation data found for this object.")

    return torch.from_numpy(mask).long()


def collate_fn(batch):
    images, targets = zip(*batch)
    images = [img for img in images]
    target_masks = [create_segmentation_mask(t) for t in targets]
    return torch.stack(images), torch.stack(target_masks)

# 加载数据集
train_dataset = CocoDetection(root=os.path.join(data_dir, "train2017"),
                               annFile=train_ann_file,
                               transform=transform)

val_dataset = CocoDetection(root=os.path.join(data_dir, "val2017"),
                             annFile=val_ann_file,
                             transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

# 定义分割模型
class SegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SegmentationModel, self).__init__()
        self.backbone = nn.Sequential(*list(resnet34(pretrained=True).children())[:-2])  # 去掉最后的全连接层
        self.conv1 = nn.Conv2d(512, num_classes, kernel_size=1)  # 分割头

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv1(x)
        return x

# PyTorch Lightning 模型
class SegmentationLightningModel(pl.LightningModule):
    def __init__(self, num_classes):
        super(SegmentationLightningModel, self).__init__()
        self.model = SegmentationModel(num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.model(images)
        outputs = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        print("targets.min(), targets.max()", targets.min(), targets.max())
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)

        pred_masks = torch.argmax(outputs, dim=1)  # 获取类别预测
        targets = targets.squeeze(1)  # 去掉通道维度以匹配形状

        # 计算 IoU
        intersection = (pred_masks == targets).float().sum((1, 2))  # 每个样本的交集
        union = (pred_masks + targets > 0).float().sum((1, 2))  # 每个样本的并集

        # 避免除以零
        iou = intersection / (union + 1e-6)  # 加上小值避免除以零

        mean_iou = iou.mean()  # 计算平均 IoU

        # 记录损失和 IoU
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)
        self.log('mean_iou', mean_iou)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

# 训练模型
num_classes = 91  # COCO 数据集中的类别数量
model = SegmentationLightningModel(num_classes)

trainer = pl.Trainer(max_epochs=10, devices=[1], accelerator='gpu')  # 使用合适的设备
trainer.fit(model, train_loader, val_loader)
