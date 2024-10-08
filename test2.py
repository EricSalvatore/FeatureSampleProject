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
        pro_logits = self.pro1(logits).unsqueeze(1).view(-1, 1, 112, 112)
        return logits, self.seg_head(pro_logits)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images, targets = images.to(self._device), targets.to(self._device)
        _, outputs = self(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss)
        print(f"train_loss is {loss:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images, targets = images.to(self._device), targets.to(self._device)
        _, outputs = self(images)
        outputs = F.interpolate(outputs, size=(224, 224), mode='bilinear', align_corners=False)
        loss = F.cross_entropy(outputs, targets)
        self.log('val_loss', loss)
        print(f"val_loss is {loss:.4f}")

        pred_masks = torch.argmax(outputs, dim=1)
        targets = targets.squeeze(1)

        # 计算 IoU
        intersection = (pred_masks == targets).float().sum((1, 2))
        union = ((pred_masks != 0) | (targets != 0)).float().sum((1, 2))
        iou = intersection / (union + 1e-6)
        mean_iou = iou.mean()

        self.log('mean_iou', mean_iou)
        self.avg_acc(mean_iou)
        print('mean_iou: ', mean_iou)
        print('val_loss: ', loss)

    def on_validation_epoch_end(self):
        avg_acc = self.avg_acc.compute()
        self.log("avg_acc", avg_acc)
        print(f"avg acc is {avg_acc}")
        self.avg_acc.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)


def create_segmentation_mask(target, image_size=(224, 224), device='cuda'):
    mask = torch.zeros(image_size, dtype=torch.uint8, device=device)

    for obj in target:
        if 'segmentation' in obj:
            segmentation = obj['segmentation']
            if isinstance(segmentation, list):
                for seg in segmentation:
                    polygon = torch.tensor(seg, dtype=torch.int32, device=device).reshape((-1, 1, 2))
                    cv2.fillPoly(mask.cpu().numpy(), polygon.cpu().numpy(), color=obj['category_id'])
            elif isinstance(segmentation, dict) and 'counts' in segmentation:
                rle = maskUtils.frPyObjects(segmentation, image_size[0], image_size[1])
                rle_mask = maskUtils.decode(rle)
                rle_mask_resized = torch.tensor(cv2.resize(rle_mask, image_size[::-1], interpolation=cv2.INTER_NEAREST),
                                                device=device)
                mask = torch.maximum(mask, rle_mask_resized)

    return mask.long()


def collate_fn(batch):
    images, targets = zip(*batch)
    images = [img.to('cuda') for img in images]  # 确保将图像移到GPU
    target_masks = [create_segmentation_mask(t) for t in targets]
    return torch.stack(images), torch.stack(target_masks).to('cuda')


def train(args, _device, _train_loader, _val_loader):
    pretrained_model, process = clip.load(args.clip_image_name, device=_device)
    pretrained_vit_model = pretrained_model.visual.float()
    model = SegmentationLightningModel(num_classes=args.num_classes, _model_features=pretrained_vit_model,
                                       _device=_device)

    checkpoint_callback = ModelCheckpoint(
        monitor='avg_acc',
        dirpath='./clip_seg_checkpoint',
        filename=f'best-{args.model_name}-checkpoint-epoch-{args.pretrained_epoch}-best-acc-{{avg_acc:.3f}}',
        save_top_k=1,
        mode='max'
    )

    trainer = Trainer(max_epochs=args.pretrained_epoch, devices=[4], accelerator='gpu', callbacks=[checkpoint_callback])
    trainer.fit(model, _train_loader, _val_loader)


if __name__ == '__main__':
    # ... 添加您的参数配置代码 ...
    parse = argparse.ArgumentParser(description="coco")
    parse.add_argument("--model_name", default='resnet18', type=str, help="model name")
    parse.add_argument("--clip_image_name", default='ViT-B/32', type=str, help="CLIP image model name")
    parse.add_argument("--data_dir", default='./data/coco1000/coco2017_1000', type=str, help="CLIP image model name")

    parse.add_argument("--train_ann_file", default='annotations/instances_train2017.json', type=str,
                       help="train ann path")
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
    mp.set_start_method('spawn', force=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl.seed_everything(42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CocoDetection(root=os.path.join(args.data_dir, "train2017"), annFile=os.path.join(args.data_dir, args.train_ann_file), transform=transform)
    val_dataset = CocoDetection(root=os.path.join(args.data_dir, "val2017"),
                                annFile=os.path.join(args.data_dir, args.val_ann_file), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                              collate_fn=collate_fn, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8,
                            collate_fn=collate_fn, pin_memory=False)

    if args.train:
        train(args, device, train_loader, val_loader)
