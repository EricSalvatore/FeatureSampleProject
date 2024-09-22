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
    parse.add_argument("--batch_size", default=2, type=int, help="batch size")
    parse.add_argument("--train", default=False, type=bool, help="whether is train process")
    parse.add_argument("--checkpoint", default='./checkpoint', type=str, help="checkpoint path")

    args = parse.parse_args()

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

    model = CIFAR10Classifier()
    # 获取已经保存好的模型
    model.load_from_checkpoint("./")

    #  获取模型 计算与类别C最相似的类别Y
    ##
    # model.eval()
    #
    # ## 保存每一个样本的pred和label
    # all_pred = []
    # all_label = []
    #
    # with torch.no_grad():
    #     for image, label in cifar10_train_dataloader:
    #         pred, logits = model(image)
    #         all_label.append(label)
    #         all_pred.append(pred)
    #
    #
    # print("")



    # # 使用预训练的ResNet-18模型
    # model = models.resnet18(pretrained=True)
    # model.eval()
    #
    # # 2. 保存每个样本的logit和标签
    # all_logits = []
    # all_labels = []
    #
    # with torch.no_grad():
    #     for images, labels in data_loader:
    #         logits = model(images)  # 模型输出的logits
    #         all_logits.append(logits)
    #         all_labels.append(labels)
    #
    # # 将所有batch的logit和标签合并到一个tensor中
    # all_logits = torch.cat(all_logits, dim=0)  # shape: (num_samples, num_classes)
    # all_labels = torch.cat(all_labels, dim=0)  # shape: (num_samples, )
    #
    # # 3. 计算每个类别的平均预测分数
    # num_classes = 10  # CIFAR-10有10个类别
    # average_scores = torch.zeros(num_classes, num_classes)  # shape: (num_classes, num_classes)
    #
    # # 遍历每个类别，计算该类别的平均logit分数
    # for c in range(num_classes):
    #     class_mask = (all_labels == c)  # 当前类别的mask
    #     class_logits = all_logits[class_mask]  # 当前类别的所有logit
    #     class_avg_scores = class_logits.mean(dim=0)  # 计算所有logit的平均值
    #     average_scores[c] = class_avg_scores
    #
    # # 4. 寻找最相似的类别（除了当前类别之外，平均logit分数最大的类别）
    # most_similar_classes = []
    #
    # for c in range(num_classes):
    #     avg_scores_c = average_scores[c]  # 当前类别的平均logit分数
    #     # 排除当前类别，找出其他类别中logit分数最大的类别
    #     avg_scores_c[c] = float('-inf')  # 当前类别置为负无穷大，不参与比较
    #     most_similar_class = torch.argmax(avg_scores_c).item()  # 最相似类别
    #     most_similar_classes.append(most_similar_class)
    #     print(f"Class {c} is most similar to class {most_similar_class}")
    #
    # # 输出每个类别最相似的类别
    # print("Most similar classes for each class:", most_similar_classes)