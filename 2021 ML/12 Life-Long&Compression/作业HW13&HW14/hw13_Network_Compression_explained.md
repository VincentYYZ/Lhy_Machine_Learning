# hw13_Network_Compression 逐行讲解版

面向零基础同学，对 `hw13_Network_Compression.ipynb` 代码逐段解释，涵盖模型压缩（小模型设计、知识蒸馏、伪标签半监督）流程。按 notebook 顺序讲解。

## 数据下载与加载（同 HW3）

```python
!gdown --id '1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy' --output food-11.zip
!unzip -q food-11.zip
```
- 下载/解压 food-11 数据集（11 类食物，含 labeled/validation/unlabeled/testing）。

### 导入包

```python
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms, torchvision.models as models
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
```
- 主要依赖 torch/torchvision、PIL、数据加载与进度条。

### 数据增强与 DataLoader

```python
train_tfm = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomCrop(128),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])
batch_size = 64
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set , batch_size=batch_size, shuffle=False)
```
- 训练增强更强；验证/测试只缩放+中心裁剪。batch=64。

## 小模型设计（< 100k 参数）

```python
class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 100, 3), nn.BatchNorm2d(100), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1,1)),  # GAP 适配可变尺寸
        )
        self.fc = nn.Sequential(nn.Linear(100, 11))
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)
```
- TODO 可自行改用 depthwise/pointwise/group conv 减参，但需小于 100k 参数。示例参数约 88k。

### 模型统计

```python
from torchsummary import summary
student_net = StudentNet()
summary(student_net, (3, 128, 128), device="cpu")
```
- 输出层级与参数量，需提交截图/数字。

## 知识蒸馏损失

```python
def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5, T=1.0):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    # TODO: 完成 soft loss
    # soft_loss = alpha * T*T * F.kl_div(
    #     F.log_softmax(outputs/T, dim=1),
    #     F.softmax(teacher_outputs/T, dim=1),
    #     reduction='batchmean')
    soft_loss = 0
    return hard_loss + soft_loss
```
- 硬标签交叉熵 + 软标签 KL，软部分需按公式补全（记得乘 T^2）。

## 加载教师模型

```python
!gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output teacher_net.ckpt
teacher_net = torch.load('./teacher_net.ckpt')
teacher_net.eval()
```
- 提供的 ResNet 教师；如果修改数据预处理需确保与教师兼容。

## 伪标签半监督

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
student_net = student_net.to(device); teacher_net = teacher_net.to(device)
do_semi = True

def get_pseudo_labels(dataset, model):
    loader = DataLoader(dataset, batch_size=batch_size*3, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for img, _ in tqdm(loader):
        with torch.no_grad():
            logits = model(img.to(device))
            pseudo_labels.append(logits.argmax(dim=-1).cpu())
    pseudo_labels = torch.cat(pseudo_labels)
    # 替换 DatasetFolder 的 samples 标签
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset

if do_semi:
    unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
    concat_dataset = ConcatDataset([train_set, unlabeled_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
```
- 用教师给 unlabeled 集打标签，合并到训练集，形成新的 DataLoader。

## 训练（蒸馏版，同 HW3 框架）

```python
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(student_net.parameters(), lr=0.0003, weight_decay=1e-5)
n_epochs = 80
for epoch in range(n_epochs):
    student_net.train()
    train_loss, train_accs = [], []
    for imgs, labels in tqdm(train_loader):
        logits = student_net(imgs.to(device))
        with torch.no_grad():
            soft_labels = teacher_net(imgs.to(device))
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)
        optimizer.zero_grad(); loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item()); train_accs.append(acc)
    print(f"[ Train | {epoch+1:03d}/{n_epochs:03d} ] loss = {sum(train_loss)/len(train_loss):.5f}, acc = {sum(train_accs)/len(train_accs):.5f}")

    student_net.eval()
    valid_loss, valid_accs = [], []
    for imgs, labels in tqdm(valid_loader):
        with torch.no_grad():
            logits = student_net(imgs.to(device))
            soft_labels = teacher_net(imgs.to(device))
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().cpu().numpy()
        valid_loss.append(loss.item()); valid_accs += list(acc)
    print(f"[ Valid | {epoch+1:03d}/{n_epochs:03d} ] loss = {sum(valid_loss)/len(valid_loss):.5f}, acc = {sum(valid_accs)/len(valid_accs):.5f}")
```
- 与 HW3 类似，但损失使用蒸馏版本；训练/验证循环分别输出均值 loss/acc。可调整 epoch/学习率等。

## 测试与提交

```python
student_net.eval()
predictions = []
for imgs, _ in tqdm(test_loader):
    with torch.no_grad():
        logits = student_net(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in enumerate(predictions):
        f.write(f"{i},{pred}\n")
```
- 推理测试集，写出提交文件。

## 关键点与改进建议
- 补全 `loss_fn_kd` 的软损失（KL+T^2）。
- 设计更高效的小模型：使用 depthwise/pointwise/group conv、SE、较小宽度等，确保参数 <100k。
- 半监督伪标签阈值：可过滤低置信度样本或仅用 top-k。
- 训练技巧：余弦/Step LR 调度、Label Smoothing、Mixup/CutMix（需与教师兼容）、梯度累积。

> 以上逐段解释 notebook 主体代码，帮助理解网络压缩作业的模型设计、蒸馏与伪标签流程。祝训练顺利！***
