# HW14_lifelong_learning 逐行讲解版

面向零基础同学，对 `HW14_lifelong_learning.ipynb` 逐段解释，涵盖基于 permuted MNIST 的连续学习实验与若干正则化方法（EWC/MAS/SI/RWalk/SCP）的框架。按 notebook 顺序讲解。

## 背景
- 终身学习/连续学习：模型依次学习多任务，避免遗忘旧任务。
- 本作业使用 permuted MNIST：对 MNIST 像素置乱生成多任务（每个任务一个固定置乱）。
- 重点在比较/实现不同 regularization-based 方法。

## 数据与工具

### 导入/置乱

```python
import torch.utils.data as data, torch.utils.data.sampler as sampler
import torchvision, os, torch.nn.functional as F
from torchvision import datasets, transforms

def _permutate_image_pixels(image, permutation):
    if permutation is None: return image
    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image
```
- `_permutate_image_pixels` 按给定 permutation 重排像素（784 维）。

```python
def get_transform(permutation=None, normalize=True):
    tfms = [transforms.ToTensor(), Pad(28)]
    if normalize: tfms.append(transforms.Normalize((0.1307,), (0.3081,)))
    tfms.append(transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)))
    return transforms.Compose(tfms)
```
- 置乱前先 ToTensor + 可选 Normalize + Pad 到 28x28（居中填充）。

```python
class Pad(object):
    def __init__(self, size, fill=0, padding_mode='constant'): ...
    def __call__(self, img):
        img_size = img.size()[1]
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)
```
- 辅助类，填充到指定尺寸。

```python
class Data():
    def __init__(self, path, train=True, permutation=None, normalize=True):
        transform = get_transform(permutation, normalize)
        self.dataset = datasets.MNIST(root=os.path.join(path, "MNIST"),
                                      transform=transform,
                                      train=train,
                                      download=True)
```
- 包装 MNIST 数据集，应用置乱。

### 任务/数据加载器

```python
class Args:
    task_number = 5
    epochs_per_task = 10
    lr = 1.0e-4
    batch_size = 128
    test_size = 8192
    random_seed = 0
args = Args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(args.random_seed)
permutations = [np.random.permutation(784) if i !=0 else np.arange(784) for i in range(args.task_number)]
train_datasets = [Data('data', permutation=perm) for perm in permutations]
train_dataloaders = [DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets]
test_datasets = [Data('data', train=False, permutation=perm) for perm in permutations]
test_dataloaders = [DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets]
```
- 生成 5 个任务的置乱；构建对应 train/test DataLoader。

## 网络（示例）

通常是简单的 MLP/CNN（notebook 中省略定义位置，此处提醒）：
- 输入维度 28x28=784（置乱后展平）或 1x28x28。
- 输出 10 类。
- 需要对不同方法计算/存储重要性或路径累积等。

## 方法提示

作业涵盖以下正则化思路（实现细节在 notebook 下文）：
- **EWC**：计算 Fisher 信息（基于旧任务数据的梯度平方期望），对参数偏离旧参数加惩罚。
- **MAS**：用输出 L2 范数梯度近似 Fisher，得到重要性。
- **SI**：记录优化过程中每步参数变化与梯度内积，得到路径重要性（ω），训练结束累积到 Ω。
- **RWalk**：结合 SI 的路径正则与临近惩罚。
- **SCP**：基于敏感度的正则（notebook 包含实现框架）。

### 训练循环框架

通常如下：
```python
for task_id in range(args.task_number):
    # 训练当前任务 epochs_per_task 轮
    for epoch in range(args.epochs_per_task):
        for x, y in train_dataloaders[task_id]:
            # 前向/损失 = 当前任务 CE + 正则(对旧任务的重要性约束)
            # 反传/优化
    # 任务结束后，计算并存储重要性/Fisher/Ω 等，用于后续任务正则
```
- 正则项依赖于前面任务的参数快照 θ* 和重要性估计。

### 评估

```python
for task_id in range(args.task_number):
    acc = eval(model, test_dataloaders[task_id])
    print(...)
```
- 记录各任务测试精度，观察遗忘与保持情况。

## 可视化（若有）

notebook 可能包含 loss/acc 曲线或混淆矩阵；使用 matplotlib 绘制。

## 关键实现点
- 保存每个任务结束时的参数副本（θ_prev）。
- 重要性计算（EWC: Fisher；MAS: 梯度输出范数；SI: 路径累积）与正则公式。
- 正则系数 λ 超参控制稳定/可塑性平衡。
- Optimizer/调度可按需调整；batch_size/epoch 在 Args 中设定。

> 以上为 notebook 核心代码逻辑的逐行说明，涵盖 permuted MNIST 数据准备、任务加载、连续学习正则方法框架与训练/评估流程。具体算法公式请对照课件/论文在 TODO 处补全实现。祝实验顺利！***
