# ML2021_HW15_Meta_Learning 逐行讲解版

面向零基础同学，对 `ML2021_HW15_Meta_Learning.ipynb` 逐段解释，帮助理解在 Omniglot 上实现 MAML（少样本分类）。按 notebook 顺序讲解。

## 环境与数据

```python
!nvidia-smi
try: import qqdm
except: !pip install qqdm > /dev/null 2>&1
```
- 查看 GPU；安装进度条 qqdm。

### 下载与解压 Omniglot

```python
workspace_dir = '.'
!gdown --id 1FLDrQ0k-iJ-mk8ors0WItqvwgu0w9J0U --output "{workspace_dir}/Omniglot.tar.gz"
!tar -zxf "{workspace_dir}/Omniglot.tar.gz" -C "{workspace_dir}/"
```
- 下载预处理好的 Omniglot 数据并解压到当前目录。

### 数据预览

```python
from PIL import Image
for i in range(10,20):
    im = Image.open("Omniglot/images_background/Japanese_(hiragana).0/character13/0500_" + str(i) + ".png")
    display(im)
```
- 展示若干样本，字符大小为 28x28（灰度）。

## 导入库与随机种子

```python
import glob, random
from collections import OrderedDict
import numpy as np
from qqdm.notebook import qqdm as tqdm  # 若无则用 tqdm.auto
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
random_seed = 0
random.seed(random_seed); np.random.seed(random_seed); torch.manual_seed(random_seed)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(random_seed)
```
- 设置环境与种子以保证可复现。

## 模型模块（支持 functional forward）

MAML 需要在内循环使用“功能式”前向传播（显式传入权重），以便对原始参数求梯度。

```python
def ConvBlock(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )

def ConvBlockFunction(x, w, b, w_bn, b_bn):
    x = F.conv2d(x, w, b, padding=1)
    x = F.batch_norm(x, running_mean=None, running_var=None,
                     weight=w_bn, bias=b_bn,
                     training=True)
    x = F.relu(x)
    x = F.max_pool2d(x, 2, 2)
    return x
```
- `ConvBlock` 正常模块；`ConvBlockFunction` 用于 functional forward，显式传入卷积/BN 权重。

### 四层 CNN（MAML 常用）

```python
class Learner(nn.Module):
    def __init__(self, n_class=5):
        super().__init__()
        self.features = nn.ModuleList([
            ConvBlock(1, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
            ConvBlock(64, 64),
        ])
        self.classifier = nn.Linear(64, n_class)

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def functional_forward(self, x, params):
        x = ConvBlockFunction(x, params['features.0.0.weight'], params['features.0.0.bias'],
                                 params['features.0.1.weight'], params['features.0.1.bias'])
        x = ConvBlockFunction(x, params['features.1.0.weight'], params['features.1.0.bias'],
                                 params['features.1.1.weight'], params['features.1.1.bias'])
        x = ConvBlockFunction(x, params['features.2.0.weight'], params['features.2.0.bias'],
                                 params['features.2.1.weight'], params['features.2.1.bias'])
        x = ConvBlockFunction(x, params['features.3.0.weight'], params['features.3.0.bias'],
                                 params['features.3.1.weight'], params['features.3.1.bias'])
        x = x.view(x.size(0), -1)
        x = F.linear(x, params['classifier.weight'], params['classifier.bias'])
        return x
```
- 标准 4 层卷积 + 全连接，`functional_forward` 按命名参数 dict 前向。

## 数据集定义（Few-shot Episode）

```python
class Omniglot(Dataset):
    def __init__(self, root, mode, n_way=5, k_shot=1, k_query=15, resize=28):
        self.n_way, self.k_shot, self.k_query = n_way, k_shot, k_query
        self.resize = resize
        # 读取背景/评估集目录，构造类别列表
        if mode == 'train':
            self.meta = glob.glob(root + '/images_background/*')
        else:
            self.meta = glob.glob(root + '/images_evaluation/*')
        # 每个类的所有图片路径
        self.cl_list = []
        for cl in self.meta:
            self.cl_list.append(glob.glob(cl + '/*/*png'))
        self.n_cls = len(self.cl_list)
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.922,), std=(0.084,)),  # 预计算的均值/方差
        ])
```
- 划分 train/eval 集；预存每类所有图片路径；定义灰度+缩放+归一化的 transform。

### 抽取 episode

```python
    def __getitem__(self, idx):
        # 随机抽 n_way 类
        selected_cls = np.random.choice(self.n_cls, self.n_way, replace=False)
        support_x = []; query_x = []; support_y = []; query_y = []
        for i, cls in enumerate(selected_cls):
            selected_imgs = np.random.choice(self.cl_list[cls], self.k_shot + self.k_query, replace=False)
            support_imgs = selected_imgs[:self.k_shot]
            query_imgs = selected_imgs[self.k_shot:]
            support_x += support_imgs; query_x += query_imgs
            support_y += [i]*self.k_shot; query_y += [i]*self.k_query
        random.shuffle(support_x); random.shuffle(query_x)
        support_x = torch.stack([self.transform(Image.open(x)) for x in support_x])
        query_x   = torch.stack([self.transform(Image.open(x)) for x in query_x])
        support_y = torch.tensor(support_y)
        query_y   = torch.tensor(query_y)
        return support_x, support_y, query_x, query_y

    def __len__(self):
        return 1000000  # 虚拟长度，按需要采样
```
- 每次返回一个 episode：n_way*(k_shot+k_query) 张图片及标签。

## 超参数与 MAML 设置

```python
n_way = 5; k_shot = 1; k_query = 15
meta_batch_size = 2    # 每个 meta step 处理的 episode 数
inner_step = 5         # 内循环梯度步数
inner_lr = 0.4
meta_lr = 1e-3
epochs = 30
train_db = Omniglot('Omniglot', mode='train', n_way=n_way, k_shot=k_shot, k_query=k_query)
test_db  = Omniglot('Omniglot', mode='test',  n_way=n_way, k_shot=k_shot, k_query=k_query)
train_loader = DataLoader(train_db, meta_batch_size, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_db, meta_batch_size, shuffle=False, num_workers=2)
```
- 定义 5-way 1-shot 15-query 设置及内/外学习率等。

## MAML 训练核心

```python
def clone_state_dict(model):
    return OrderedDict((name, p.clone()) for name, p in model.named_parameters())

model = Learner(n_class=n_way).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)

for epoch in range(epochs):
    for step, (support_x, support_y, query_x, query_y) in enumerate(tqdm(train_loader)):
        # support/query shape: [meta_batch, n_way*k, C, H, W]
        task_losses = []; task_accs = []
        for i in range(meta_batch_size):
            fast_weights = clone_state_dict(model)
            # inner loop: K steps
            for _ in range(inner_step):
                logits = model.functional_forward(support_x[i].to(device), fast_weights)
                loss = F.cross_entropy(logits, support_y[i].to(device))
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                fast_weights = OrderedDict((name, param - inner_lr*grad)
                                            for ((name, param), grad) in zip(fast_weights.items(), grads))
            # eval on query with adapted params
            query_logits = model.functional_forward(query_x[i].to(device), fast_weights)
            query_loss = F.cross_entropy(query_logits, query_y[i].to(device))
            task_losses.append(query_loss)
            task_accs.append((query_logits.argmax(dim=1) == query_y[i].to(device)).float().mean())
        # outer loop update
        meta_loss = torch.stack(task_losses).mean()
        optimizer.zero_grad()
        meta_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, meta loss {meta_loss.item():.4f}, acc {torch.stack(task_accs).mean().item():.4f}")
```
- 对每个任务复制参数（快照），在支持集上内更新若干步，再在查询集上计算损失，平均后对原始参数做外更新。

## 测试

```python
def eval(model, loader):
    model.eval()
    accs = []
    with torch.no_grad():
        for support_x, support_y, query_x, query_y in loader:
            fast_weights = clone_state_dict(model)
            for _ in range(inner_step):
                logits = model.functional_forward(support_x[0].to(device), fast_weights)
                loss = F.cross_entropy(logits, support_y[0].to(device))
                grads = torch.autograd.grad(loss, fast_weights.values())
                fast_weights = OrderedDict((name, param - inner_lr*grad)
                                            for ((name, param), grad) in zip(fast_weights.items(), grads))
            query_logits = model.functional_forward(query_x[0].to(device), fast_weights)
            accs.append((query_logits.argmax(dim=1) == query_y[0].to(device)).float().mean())
    return torch.stack(accs).mean().item()
```
- 对测试任务执行一次内更新，再评估查询集准确率。

## 关键点与改进
- 内循环步数/学习率、meta_lr、batch 大小对性能敏感，可调。
- 支持更多 layer_norm/BN 处理；当前 functional BN 使用 `training=True` 不跟踪统计量。
- 可尝试 ANIL、Meta-SGD、更深 backbone，或增大 meta_batch，提高表现。

> 以上逐行解释 notebook 核心部分，涵盖数据准备、模型（功能式前向）、MAML 内外循环训练与测试流程，帮助理解少样本元学习实现。祝实验顺利！***
