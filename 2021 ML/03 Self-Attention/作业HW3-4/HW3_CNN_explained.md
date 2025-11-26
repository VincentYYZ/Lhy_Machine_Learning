# HW3_CNN 逐行讲解版

面向零基础同学，对 `HW3_CNN.ipynb` 中的代码逐段逐行解释，帮助理解用 CNN 做食物图像 11 类分类，以及半监督伪标签逻辑。代码块后紧跟解释，顺序与 notebook 一致。

## 下载与解压数据集

```python
# Google Drive
# !gdown --id '1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy' --output food-11.zip

# Dropbox
!wget https://www.dropbox.com/s/m9q6273jl3djall/food-11.zip -O food-11.zip

# Unzip the dataset.
!unzip -q food-11.zip
```
- 注释的 gdown：备用从 GDrive 下载。
- `wget ... -O food-11.zip`：从 Dropbox 下载压缩包并命名。
- `!unzip -q`：静默解压得到 `food-11/`。

## 导入依赖

```python
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
```
- NumPy 用于数值操作；PyTorch 核心与神经网络模块。
- `torchvision.transforms`：图像变换与数据增强。
- `PIL.Image`：读取图像。
- `ConcatDataset/Subset/DataLoader`：组合、子集、批量加载数据。
- `DatasetFolder`：按目录结构（子文件夹为类名）自动加载图片。
- `tqdm`：进度条。

## 图像变换与数据集加载

```python
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    # 这里可插入更多增强
    transforms.ToTensor(),
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
```
- `Compose` 串联变换；训练集统一缩放到 128x128，可再加随机翻转/裁剪等增强；`ToTensor` 把 PIL 转为张量并归一化到 [0,1]。

```python
batch_size = 128

train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```
- `DatasetFolder`：按目录名作为标签读取 `jpg`，用指定 loader 打开图片并应用变换。
- `batch_size=128`：每批 128 张。
- 训练/验证打乱；测试不打乱；`pin_memory=True` 便于 GPU 拷贝。

## 定义 CNN 模型

```python
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)   # 卷积提特征 [B,256,8,8]
        x = x.flatten(1)         # 展平为 [B,256*8*8]
        x = self.fc_layers(x)    # 全连接得到 11 类 logits
        return x
```
- 三段卷积+BN+ReLU+池化；输入 3 通道，通道逐层 64→128→256，空间逐步减半/四分。
- 全连接两层 256 隐层，最后输出 11 类。
- 前向：卷积→展平→全连接，输出未做 softmax 的 logits。

## 半监督伪标签函数（需完成 TODO）

```python
def get_pseudo_labels(dataset, model, threshold=0.65):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    softmax = nn.Softmax(dim=-1)
    for batch in tqdm(dataloader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)
        # TODO: 根据 probs 过滤高置信度样本，构建带伪标签的新 DatasetFolder
    model.train()
    return dataset
```
- 设 eval 模式、softmax 得到概率。
- 遍历未标记集，计算概率；需筛选 `max_prob >= threshold` 的样本并生成带伪标签数据集，再返回。
- 结束后恢复 train 模式。

## 训练主循环

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
model.device = device
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
n_epochs = 80
do_semi = False
```
- 选择设备，实例化模型；交叉熵损失；Adam 优化；最多 80 轮；`do_semi` 控制是否使用伪标签。

```python
for epoch in range(n_epochs):
    if do_semi:
        pseudo_set = get_pseudo_labels(unlabeled_set, model)
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    model.train()
    train_loss, train_accs = [], []
    for batch in tqdm(train_loader):
        imgs, labels = batch
        logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        train_loss.append(loss.item()); train_accs.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
```
- 若启用半监督：用当前模型给未标记数据打伪标签后，与有标记集合并再加载。
- 训练阶段：前向→交叉熵→清梯度→反传→梯度裁剪（防爆炸）→更新；计算 batch 准确率；记录平均损失/准确率并打印。

```python
    model.eval()
    valid_loss, valid_accs = [], []
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad():
          logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        valid_loss.append(loss.item()); valid_accs.append(acc)

    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
```
- 验证：eval 模式 + `torch.no_grad()`；计算平均损失与准确率并打印。

## 测试与提交文件

```python
model.eval()
predictions = []
for batch in tqdm(test_loader):
    imgs, labels = batch  # labels 在测试集是占位 0
    with torch.no_grad():
        logits = model(imgs.to(device))
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
```
- 推理：关闭梯度，取每张图片 logits 最大的类别，收集预测。

```python
with open("predict.csv", "w") as f:
    f.write("Id,Category\n")
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
```
- 写出提交文件，第一行表头，后续每行对应图片索引与预测类别。

> 以上覆盖 notebook 所有代码行，便于理解数据加载、模型、训练、验证、测试以及可选伪标签流程。记得若要半监督需补完 `get_pseudo_labels` 的 TODO。祝训练顺利！
