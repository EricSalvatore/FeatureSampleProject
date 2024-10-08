import argparse
import os
import torch
from torch.utils.data import DataLoader

from resne_seg import SegmentationLightningModel
from torchvision import models
from torchvision.datasets import CocoDetection
from torchvision import transforms
import numpy as np
import cv2
import torch.nn.functional as F
from pycocotools import mask as maskUtils
import torch.multiprocessing as mp


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

pretrained_model = models.resnet18(pretrained=True)
model = SegmentationLightningModel(num_classes=91, _model_features=pretrained_model)

checkpoint = torch.load('./seg_res_checkpoint/best-resnet18-checkpoint-epoch-100-best-acc-avg_acc=8874673152.000-curepoch-epoch=7.ckpt')
# 只加载状态字典
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 设置为评估模式

print(model)

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
    return torch.stack(images).to('cuda'), torch.stack(target_masks).to('cuda')

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

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=False)

from PIL import Image
output_dir = r"./res_seg_img"
model = model.to('cuda')
with torch.no_grad():  # 禁用梯度计算
    for i, (images, targets) in enumerate(val_dataset):
        inputs = images.unsqueeze(0).cuda()
        # 进行推理
        _, outputs = model(inputs)
        output = F.interpolate(outputs, size=(256, 256), mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(output, dim=1)
        print(pred_mask.max())
        pred_mask = pred_mask.squeeze(0)

        seg_map_image = Image.fromarray((pred_mask.cpu().numpy() * 255).astype(np.uint8))
        # 保存输出图像
        seg_map_image.save(os.path.join(output_dir, f'segmentation_{i}.png'))


