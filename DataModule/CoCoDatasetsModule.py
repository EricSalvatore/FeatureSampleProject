import pytorch_lightning as pl
from torchvision.datasets import CocoDetection
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CocoDataModule(pl.LightningDataModule):
    def __init__(self, train_img_dir, train_ann_file, val_img_dir, val_ann_file, batch_size=16):
        super().__init__()
        self.train_img_dir = train_img_dir
        self.train_ann_file = train_ann_file
        self.val_img_dir = val_img_dir
        self.val_ann_file = val_ann_file
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.target_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # 可以添加其他变换，例如转换为张量等
        ])

    def setup(self, stage=None):
        self.train_dataset = CocoDetection(root=self.train_img_dir, annFile=self.train_ann_file, transform=self.transform, target_transform=self.target_transform)
        self.val_dataset = CocoDetection(root=self.val_img_dir, annFile=self.val_ann_file, transform=self.transform, target_transform=self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
