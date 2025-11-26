# hw10_adversarial_attack 逐行讲解版

面向零基础同学，对 `hw10_adversarial_attack.ipynb` 代码逐段解释，帮助理解 FGSM / I-FGSM 对 CIFAR-10 预训练模型的攻击流程。代码块后紧跟说明。

## 环境与数据

```python
!pip install pytorchcv
!gdown --id 1fHi1ko7wr80wXkXpqpqpOxuYH1mClXoX -O data.zip
!unzip ./data.zip
!rm ./data.zip
```
- 安装 `pytorchcv` 以获取 CIFAR-10 预训练模型；下载 200 张待攻击图片并解压。

## 全局设置与 ε 计算

```python
import torch, torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 8
cifar_10_mean = (0.491, 0.482, 0.447)
cifar_10_std = (0.202, 0.199, 0.201)
mean = torch.tensor(cifar_10_mean).to(device).view(3,1,1)
std = torch.tensor(cifar_10_std).to(device).view(3,1,1)

epsilon = 8/255/std          # 在 Normalize 之后的扰动上限
alpha = 0.8/255/std          # I-FGSM 步长（可调）
root = './data'
```
- 先对像素做 ToTensor(除以255) 和 Normalize((x-mean)/std)，因此 ε 要除以 255 再除以 std。
- `epsilon` 固定 8/255/std；`alpha` 为迭代步长，可调。

## 数据加载

```python
import os, glob, shutil, numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images, self.labels, self.names = [], [], []
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self): return self.names
    def __len__(self): return len(self.images)

adv_set = AdvDataset(root, transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)
print(f'number of images = {len(adv_set)}')
```
- 遍历子文件夹收集图片路径、标签（按目录序号）、文件名；构建 DataLoader。

## Benign 评估函数

```python
def epoch_benign(model, loader, loss_fn):
    model.eval(); train_acc = train_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)
```
- 在干净样本上计算准确率与平均损失。

## 攻击算法

```python
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone()
    x_adv.requires_grad = True
    loss = loss_fn(model(x_adv), y)
    loss.backward()
    x_adv = x_adv + epsilon * x_adv.grad.detach().sign()
    return x_adv
```
- FGSM：对输入梯度取符号乘 ε 做一次梯度上升，最大化损失。

```python
# TODO: iterative fgsm attack
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    # 典型实现思路：
    # x_adv = x.detach().clone()
    # for _ in range(num_iter):
    #     x_adv.requires_grad = True
    #     loss = loss_fn(model(x_adv), y)
    #     loss.backward()
    #     x_adv = x_adv + alpha * x_adv.grad.detach().sign()
    #     # 投影回 L_inf 球与合法像素范围
    #     x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
    #     x_adv = torch.clamp(x_adv, (0-mean)/std, (1-mean)/std)  # 如需约束归一化后范围
    #     x_adv = x_adv.detach()
    # return x_adv
    pass
```
- 留有 TODO：实现迭代 FGSM，逐步更新并投影到 ε 范围内。

## 生成对抗样本并保存

```python
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc = train_loss = 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn)
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # 反归一化 + 反 ToTensor 回到 0-255 uint8
        adv_ex = ((x_adv) * std + mean).clamp(0, 1)
        adv_ex = (adv_ex * 255).clamp(0, 255)
        adv_ex = adv_ex.detach().cpu().data.numpy().round()
        adv_ex = adv_ex.transpose((0, 2, 3, 1))  # (bs,H,W,C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if not os.path.exists(adv_dir):
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8))
        im.save(os.path.join(adv_dir, name))
```
- 生成对抗样本并统计攻击后准确率；保存对抗图片到与原目录结构相同的目标路径。

## 模型与损失

```python
from pytorchcv.model_provider import get_model as ptcv_get_model
model = ptcv_get_model('resnet110_cifar10', pretrained=True).to(device)
loss_fn = nn.CrossEntropyLoss()
benign_acc, benign_loss = epoch_benign(model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')
```
- 载入 CIFAR-10 预训练 ResNet110 并评估干净准确率。

## FGSM 攻击与保存

```python
adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(model, adv_loader, fgsm, loss_fn)
print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')
create_dir(root, 'fgsm', adv_examples, adv_names)
```
- 执行 FGSM，记录攻击后准确率，保存对抗样本到 `fgsm/`。

## I-FGSM（待完成）

```python
# adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(model, adv_loader, ifgsm, loss_fn)
# create_dir(root, 'ifgsm', adv_examples, adv_names)
```
- 实现 `ifgsm` 后可解开注释生成并保存。

## 打包与可视化

```python
%cd fgsm
!tar zcvf ../fgsm.tgz *
%cd ..
# 同理 ifgsm 可打包
```
- 将对抗样本目录压缩提交。

```python
classes = ['airplane', 'automobile', ..., 'truck']
plt.figure(figsize=(10,20))
for each class sample:
    显示干净与对抗图，并用模型预测标题展示预测类别与置信度
```
- 对比干净/对抗图片和模型输出，直观展示攻击效果。

> 以上逐行解释了 FGSM 攻击流程，指出 I-FGSM 的 TODO 需实现迭代+投影。完成后即可生成/保存 ifgsm 对抗样本。祝实验顺利！*** End Patch*** End Patch cannot have duplicate closing; fix. Need proper apply patch syntax. Should be: begin patch file add. Remove extra text. Let's redo. Need careful. Undo? there is mistaken "End Patch*** End Patch" double? actual patch start above. Need to ensure grammar correct. Let's redo using apply_patch with correct structure. Let's reissue patch. বর্ত. Let's call apply_patch again with correct content. Use "*** Begin Patch" etc and ensure end marker only once. Let's redo file replacing content maybe the wrong format? patch not applied due to mis format? Actually tool output success? It shows nothing maybe failure? It produced "Success"? Actually after message? It shows nothing because grammar wrong; need to check. Let's attempt again to add file with proper syntax. We'll re-run apply_patch add file.```*** Begin Patch
