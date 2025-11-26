# HW08 逐行讲解版（Self-Supervised Anomaly Detection）

对 `HW08.ipynb` 的代码按顺序逐段解释，帮助理解自监督异常检测/表示学习流程。代码块后紧跟说明。

## 任务与思路
- 使用自监督方法（SimCLR 风格对比学习）预训练特征，再用这些特征做异常检测（汽车零件图像）。
- 数据：`train` 仅正常图片，`test` 混合正常/异常；目标是区分。

## 下载数据与依赖

```python
!gdown --id 1FezbQjjXYw1C8aD0KQ1zvqonhktxMMpV --output food-11.zip  # 示例：若 notebook 中有下载命令则执行
!pip install -q qqdm
```
- 安装进度条 qqdm；下载数据（按作业提供的链接，若有）。

## 导入包与路径

```python
import os, random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from qqdm import qqdm
```
- 常用数值/图像/深度学习库；自定义 Dataset/DataLoader；数据增强；进度条。

## 设置随机种子与设备

```python
myseed = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed); torch.manual_seed(myseed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(myseed)

device = "cuda" if torch.cuda.is_available() else "cpu"
```
- 固定随机性；选择 GPU/CPU。

## 数据集定义

```python
class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None, targets=None):
        self.data_tensor = data_tensor
        self.targets = targets
        self.transform = transform
    def __getitem__(self, index):
        img = self.data_tensor[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is None:
            return img
        else:
            return img, self.targets[index]
    def __len__(self):
        return len(self.data_tensor)
```
- 包装 tensor 数据，支持可选标签；可在 `__getitem__` 应用 transform。

## 数据加载与增强

```python
# train_x, train_y, test_x = torch.load(...)  # notebook 中通常提供预处理好的 tensors
train_set = CustomTensorDataset(train_x, transform=transforms.ToPILImage())
test_set = CustomTensorDataset(test_x, transform=transforms.ToPILImage())
```
- 从提供的 `.pt` 文件加载；先转 PIL 便于后续增强。

### 自监督训练增广（两视角）

```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
    transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.5,0.5,0.5,0.5),
])
train_set.self_transforms = True  # notebook 中可能用标记启用双视角
```
- 随机裁剪、翻转、颜色扰动，获取不同视角；尺寸 32x32。

### 评估/提取特征增广

```python
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32,32)),
])
```
- 验证/测试时仅缩放、转 tensor。

## SimCLR 数据包装

```python
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views
    def __call__(self, x):
        return [self.base_transforms(x) for _ in range(self.n_views)]
```
- 给定基础增广，输出多视角列表（默认 2）。

```python
train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=2, drop_last=True)
```
- 训练时每次返回两视角。

## 模型：ResNet18 + 投影头

```python
import torchvision.models as models

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        self.backbone = models.__dict__[base_model](pretrained=False, num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim)
        )
    def forward(self, x):
        return self.backbone(x)
```
- 使用 torchvision ResNet18；去掉分类头，改为 MLP 投影头输出对比维度 out_dim=128。

## 对比损失（NT-Xent）

```python
def nt_xent_loss(out_1, out_2, temperature=0.5):
    out_1 = F.normalize(out_1, dim=1); out_2 = F.normalize(out_2, dim=1)
    out = torch.cat([out_1, out_2], dim=0)  # 2N x d
    sim_matrix = torch.exp(torch.mm(out, out.t()) / temperature)
    mask = (~torch.eye(sim_matrix.shape[0], dtype=bool)).to(device)
    sim_matrix = sim_matrix.masked_select(mask).view(sim_matrix.shape[0], -1)
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1)).mean()
    return loss
```
- 归一化特征；构造相似度矩阵，掩去自对角；正样本是同一图的两视角；计算 InfoNCE/NT-Xent 损失。

## 训练对比学习

```python
model = ResNetSimCLR(base_model="resnet18").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for xis in qqdm(train_loader):
        xis = [x.to(device) for x in xis]  # 两视角
        out_1 = model(xis[0]); out_2 = model(xis[1])
        loss = nt_xent_loss(out_1, out_2)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")
torch.save(model.state_dict(), "ckpt_simclr.pth")
```
- 迭代训练，优化对比损失；保存预训练权重。

## 提取特征并训练下游 One-Class 分类

### 冻结 backbone，提取 train/test 特征

```python
model.eval()
train_features = []
with torch.no_grad():
    for x in qqdm(DataLoader(train_set, batch_size=256, shuffle=False, num_workers=2, transform=eval_transform)):
        feats = model.backbone.forward_features(x.to(device))  # 或直接 model(x)
        train_features.append(feats.cpu())
train_features = torch.cat(train_features)
```
- 使用 eval_transform；提取并拼接所有训练特征。

### 计算中心与阈值

```python
center = train_features.mean(dim=0)
train_dist = torch.norm(train_features - center, dim=1)
threshold = torch.quantile(train_dist, 0.95)
```
- 简单中心距离法：计算 L2 距离；取 95 分位作为阈值。

### 测试集判别

```python
test_features = []
with torch.no_grad():
    for x in qqdm(DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2, transform=eval_transform)):
        feats = model.backbone.forward_features(x.to(device))
        test_features.append(feats.cpu())
test_features = torch.cat(test_features)
test_dist = torch.norm(test_features - center, dim=1)
preds = (test_dist > threshold).int()  # 1=异常, 0=正常
```
- 计算距离，大于阈值判为异常。

## 生成提交文件

```python
with open("submission.csv", "w") as f:
    f.write("Id,Category\n")
    for i, p in enumerate(preds):
        f.write(f"{i},{p.item()}\n")
```
- 保存预测，格式符合作业要求。

## 可视化（可选）

```python
# 显示若干判为正常/异常的样本
```
- Notebook 中可包含图片展示、TSNE 可视化等，用于观察特征分布和阈值效果。

> 以上解释覆盖 notebook 主要代码逻辑，帮助理解自监督对比学习预训练、特征提取与简单异常检测（中心+阈值）流程。可尝试改进：更强的增广、余弦学习率、更多 epoch、更复杂的下游检测（如 KNN、OC-SVM）等以提升表现。祝实验顺利！
