# ML2021Spring_HW1 逐行讲解版

> 本文面向机器学习初学者，整理 `ML2021Spring_HW1.ipynb` 的完整训练流程。  
> 核心主线：**下载数据 → 导入库 → 构造 Dataset/DataLoader → 定义模型 → 训练/验证/测试 → 保存预测结果**。

## 目录

- [1. 任务整体理解](#1-任务整体理解)
- [2. 下载数据](#2-下载数据)
- [3. 导入包与随机种子](#3-导入包与随机种子)
- [4. 工具函数](#4-工具函数)
- [5. 数据预处理与 Dataset](#5-数据预处理与-dataset)
- [6. DataLoader](#6-dataloader)
- [7. 定义神经网络](#7-定义神经网络)
- [8. 训练、验证与测试函数](#8-训练验证与测试函数)
- [9. 超参数配置](#9-超参数配置)
- [10. 加载数据与模型](#10-加载数据与模型)
- [11. 开始训练与结果可视化](#11-开始训练与结果可视化)
- [12. 测试并保存提交文件](#12-测试并保存提交文件)
- [13. 提升思路](#13-提升思路)

---

## 1. 任务整体理解

### 1.1 这个作业在做什么

这是一个 **回归任务（Regression）**：

- **输入（Input）**：COVID-19 相关统计特征。
- **输出（Output）**：预测未来的 `tested_positive` 数值。
- **模型（Model）**：一个简单的全连接神经网络（Fully Connected Neural Network）。
- **损失函数（Loss）**：均方误差 `MSELoss`。

### 1.2 深度学习训练的通用流程

| 步骤 | 本作业对应代码 | 作用 |
|---|---|---|
| 1 | `COVID19Dataset` | 读取、切分、标准化数据 |
| 2 | `DataLoader` | 按 batch 批量加载数据 |
| 3 | `NeuralNet` | 定义神经网络结构 |
| 4 | `MSELoss` | 衡量预测值和真实值的差距 |
| 5 | `optimizer` | 根据梯度更新模型参数 |
| 6 | `train()` | 执行完整训练循环 |
| 7 | `dev()` | 在验证集上评估效果 |
| 8 | `test()` | 在测试集上生成预测 |
| 9 | `save_pred()` | 保存提交文件 |

> **关键理解**：大多数 PyTorch 模型训练都遵循这个骨架，只是数据类型、模型结构、损失函数和优化器可能不同。

---

## 2. 下载数据

```python
tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data

!gdown --id '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF' --output covid.train.csv
!gdown --id '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O' --output covid.test.csv
```

### 代码解释

| 代码 | 解释 |
|---|---|
| `tr_path = 'covid.train.csv'` | 设置训练集文件路径，训练集包含特征和真实标签。 |
| `tt_path = 'covid.test.csv'` | 设置测试集文件路径，测试集只有特征，没有真实标签。 |
| `!gdown ... --output covid.train.csv` | 用 `gdown` 从 Google Drive 下载训练集。 |
| `!gdown ... --output covid.test.csv` | 用 `gdown` 从 Google Drive 下载测试集。 |

> **注意**：`!` 是 Jupyter Notebook 的语法，表示执行 shell 命令，不是普通 Python 语法。

---

## 3. 导入包与随机种子

```python
# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

myseed = 42069  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)
```

### 3.1 导入包

| 代码 | 作用 |
|---|---|
| `import torch` | 导入 PyTorch 主库，用于张量计算和深度学习训练。 |
| `import torch.nn as nn` | 导入神经网络模块，例如 `Linear`、`ReLU`、`MSELoss`。 |
| `Dataset, DataLoader` | 自定义数据集和批量加载数据的工具。 |
| `numpy` | 处理数组和数值计算。 |
| `csv` | 读取和写入 CSV 文件。 |
| `os` | 文件夹、路径等系统操作。 |
| `matplotlib.pyplot` | 绘制学习曲线和预测散点图。 |

### 3.2 随机种子与可复现性

| 代码 | 作用 |
|---|---|
| `myseed = 42069` | 设置随机种子。数字本身不重要，重要的是固定不变。 |
| `np.random.seed(myseed)` | 固定 NumPy 的随机行为。 |
| `torch.manual_seed(myseed)` | 固定 PyTorch CPU 上的随机行为。 |
| `torch.cuda.manual_seed_all(myseed)` | 固定 PyTorch GPU 上的随机行为。 |
| `torch.backends.cudnn.deterministic = True` | 让 CuDNN 使用确定性算法。 |
| `torch.backends.cudnn.benchmark = False` | 关闭 CuDNN 自动寻找最快算法，减少不确定性。 |

> **初学者理解**：固定随机种子就像做实验时保持条件一致。这样你修改模型后，结果变化更可能来自你的修改，而不是随机运气。

---

## 4. 工具函数

### 4.1 获取计算设备

```python
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
```

| 代码 | 解释 |
|---|---|
| `torch.cuda.is_available()` | 检查是否有可用 GPU。 |
| `'cuda'` | 表示使用 NVIDIA GPU。 |
| `'cpu'` | 表示使用 CPU。 |

### 4.2 绘制学习曲线

```python
def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()
```

### 主要逻辑

| 代码 | 解释 |
|---|---|
| `loss_record['train']` | 训练过程中每个 batch 的损失。 |
| `loss_record['dev']` | 每个 epoch 后验证集的损失。 |
| `plt.plot(...)` | 画训练损失和验证损失曲线。 |
| `plt.legend()` | 显示图例。 |

> **用途**：观察模型是正常学习、过拟合还是欠拟合。

### 4.3 绘制预测值 vs 真实值

```python
def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
    ''' Plot prediction of your DNN '''
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.show()
```

### 主要逻辑

| 代码 | 解释 |
|---|---|
| `model.eval()` | 切换到评估模式。 |
| `with torch.no_grad()` | 不计算梯度，节省显存和计算。 |
| `preds.append(...)` | 收集模型预测值。 |
| `targets.append(...)` | 收集真实标签。 |
| `plt.scatter(targets, preds)` | 画真实值和预测值的散点图。 |
| `plt.plot([-0.2, lim], [-0.2, lim])` | 画理想对角线，点越靠近线越好。 |

---

## 5. 数据预处理与 Dataset

### 5.1 Dataset 完整代码

```python
class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self,
                 path,
                 mode='train',
                 target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)
        
        if not target_only:
            feats = list(range(93))
        else:
            # TODO: Using 40 states & 2 tested_positive features (indices = 57 & 75)
            pass

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]
            
            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            
            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / self.data[:, 40:].std(dim=0, keepdim=True)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)
```

### 5.2 Dataset 负责什么

| 功能 | 对应代码 |
|---|---|
| 读取 CSV | `with open(path, 'r') as fp` |
| 跳过表头和 ID 列 | `data[1:]`、`[:, 1:]` |
| 转成浮点数 | `.astype(float)` |
| 选择特征 | `feats = list(range(93))` |
| 区分训练/验证/测试 | `mode == 'train' / 'dev' / 'test'` |
| 训练集切分 | `i % 10 != 0` |
| 验证集切分 | `i % 10 == 0` |
| 转成 PyTorch Tensor | `torch.FloatTensor(...)` |
| 标准化 | `(x - mean) / std` |

### 5.3 关键变量解释

| 变量 | 含义 |
|---|---|
| `mode` | 控制当前是训练、验证还是测试。 |
| `target_only` | 是否只选择部分重要特征。 |
| `feats` | 选中的特征列索引。 |
| `target` | 真实标签，也就是模型要预测的值。 |
| `self.data` | 输入特征，形状通常是 `(样本数, 特征数)`。 |
| `self.target` | 训练/验证时的真实标签。 |
| `self.dim` | 特征数量，例如 `93` 或 `42`。 |

### 5.4 为什么要标准化

```python
self.data[:, 40:] = (self.data[:, 40:] - mean) / std
```

这叫 **Z-score 标准化**：

- **减均值**：让数据中心接近 0。
- **除标准差**：让不同特征的尺度接近。

> **意义**：避免大数值特征主导训练，让梯度更稳定，模型更容易收敛。

---

## 6. DataLoader

```python
def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader
```

### 6.1 这个函数做什么

它把两件事封装到一起：

1. 创建 `COVID19Dataset`。
2. 用 `DataLoader` 把数据包装成可迭代的 batch。

### 6.2 DataLoader 参数解释

| 参数 | 解释 |
|---|---|
| `dataset` | 数据来源，也就是 `COVID19Dataset` 实例。 |
| `batch_size` | 每次训练取多少条样本。 |
| `shuffle=(mode == 'train')` | 训练集打乱，验证/测试集不打乱。 |
| `drop_last=False` | 最后一批不足 `batch_size` 时仍然保留。 |
| `num_workers=n_jobs` | 用几个子进程加载数据。 |
| `pin_memory=True` | 使用 GPU 时可加快 CPU 到 GPU 的数据传输。 |

> **类比**：`Dataset` 像仓库，`DataLoader` 像自动打包机，每次打包一批数据给模型训练。

---

## 7. 定义神经网络

```python
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        # TODO: you may implement L2 regularization here
        return self.criterion(pred, target)
```

### 7.1 网络结构

| 层 | 输入形状 | 输出形状 | 作用 |
|---|---|---|---|
| `nn.Linear(input_dim, 64)` | `(batch, input_dim)` | `(batch, 64)` | 全连接层，把输入映射到 64 维。 |
| `nn.ReLU()` | `(batch, 64)` | `(batch, 64)` | 激活函数，引入非线性。 |
| `nn.Linear(64, 1)` | `(batch, 64)` | `(batch, 1)` | 输出一个回归预测值。 |
| `squeeze(1)` | `(batch, 1)` | `(batch,)` | 去掉多余维度，方便和标签计算损失。 |

### 7.2 `forward` 不是完整训练

`forward` 只是 **前向传播（Forward Pass）**：

```text
输入 x → 模型 self.net → 预测 pred
```

完整训练还需要：

```text
forward → loss → backward → optimizer.step()
```

---

## 8. 训练、验证与测试函数

### 8.1 训练函数 `train`

```python
def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record
```

### 8.2 一次 batch 的训练流程

| 顺序 | 代码 | 作用 |
|---|---|---|
| 1 | `optimizer.zero_grad()` | 清空上一轮梯度。 |
| 2 | `x, y = x.to(device), y.to(device)` | 把数据移动到 CPU/GPU。 |
| 3 | `pred = model(x)` | 前向传播，得到预测。 |
| 4 | `mse_loss = model.cal_loss(pred, y)` | 计算预测和真实值的差距。 |
| 5 | `mse_loss.backward()` | 反向传播，计算梯度。 |
| 6 | `optimizer.step()` | 优化器根据梯度更新参数。 |
| 7 | `loss_record['train'].append(...)` | 记录训练损失。 |

### 8.3 优化器的创建

```python
optimizer = getattr(torch.optim, config['optimizer'])(
    model.parameters(), **config['optim_hparas'])
```

如果配置里是：

```python
config['optimizer'] = 'SGD'
config['optim_hparas'] = {'lr': 0.001, 'momentum': 0.9}
```

那么它等价于：

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9
)
```

> **优化器作用**：根据梯度更新模型权重，让 loss 越来越小。

### 8.4 验证函数 `dev`

```python
def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss
```

### 8.5 测试函数 `test`

```python
def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds
```

### 8.6 训练、验证、测试的区别

| 阶段 | 是否更新参数 | 是否有标签 | 是否计算梯度 | 目的 |
|---|---|---|---|---|
| `train` | 是 | 是 | 是 | 学习模型参数 |
| `dev` | 否 | 是 | 否 | 检查泛化能力 |
| `test` | 否 | 否 | 否 | 生成最终预测 |

---

## 9. 超参数配置

```python
device = get_device()                 # get the current available device ('cpu' or 'cuda')
os.makedirs('models', exist_ok=True)  # The trained model will be saved to ./models/
target_only = False                   # TODO: Using 40 states & 2 tested_positive features

# TODO: How to tune these hyper-parameters to improve your model's performance?
config = {
    'n_epochs': 3000,                # maximum number of epochs
    'batch_size': 270,               # mini-batch size for dataloader
    'optimizer': 'SGD',              # optimization algorithm (optimizer in torch.optim)
    'optim_hparas': {                # hyper-parameters for the optimizer (depends on which optimizer you are using)
        'lr': 0.001,                 # learning rate of SGD
        'momentum': 0.9              # momentum for SGD
    },
    'early_stop': 200,               # early stopping epochs (the number epochs since your model's last improvement)
    'save_path': 'models/model.pth'  # your model will be saved here
}
```

### 参数说明

| 参数 | 解释 |
|---|---|
| `device` | 当前使用的计算设备，可能是 `cpu` 或 `cuda`。 |
| `target_only` | 是否只使用关键特征。 |
| `n_epochs` | 最大训练轮数。 |
| `batch_size` | 每次训练使用多少条样本。 |
| `optimizer` | 优化算法名称。 |
| `lr` | 学习率，控制每次参数更新的步长。 |
| `momentum` | 动量，让更新方向更稳定。 |
| `early_stop` | 验证集长时间不提升时提前停止。 |
| `save_path` | 保存最佳模型权重的位置。 |

---

## 10. 加载数据与模型

```python
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
```

| 变量 | 含义 |
|---|---|
| `tr_set` | 训练集 DataLoader。 |
| `dv_set` | 验证集 DataLoader。 |
| `tt_set` | 测试集 DataLoader。 |

```python
model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
```

| 代码 | 解释 |
|---|---|
| `tr_set.dataset.dim` | 自动读取输入特征数量。 |
| `NeuralNet(...)` | 根据特征数量创建模型。 |
| `.to(device)` | 把模型移动到 CPU 或 GPU。 |

> **自动适配**：如果特征数从 `93` 变成 `42`，`tr_set.dataset.dim` 会自动变，第一层 `Linear` 的输入维度也会跟着变。

---

## 11. 开始训练与结果可视化

### 11.1 开始训练

```python
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
```

返回值：

| 变量 | 含义 |
|---|---|
| `model_loss` | 最佳验证集 MSE。 |
| `model_loss_record` | 训练和验证损失记录，用于画图。 |

### 11.2 绘制学习曲线

```python
plot_learning_curve(model_loss_record, title='deep model')
```

学习曲线可以帮助判断：

- **正常学习**：训练 loss 和验证 loss 都下降。
- **过拟合**：训练 loss 下降，但验证 loss 上升。
- **欠拟合**：训练 loss 和验证 loss 都很高。

### 11.3 加载最佳模型并画预测图

```python
del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set
```

| 代码 | 解释 |
|---|---|
| `del model` | 删除旧模型，释放资源。 |
| `torch.load(...)` | 读取保存的最佳权重。 |
| `model.load_state_dict(ckpt)` | 把权重加载回模型。 |
| `plot_pred(...)` | 画预测值和真实值的散点图。 |

---

## 12. 测试并保存提交文件

```python
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

preds = test(tt_set, model, device)  # predict COVID-19 cases with your model
save_pred(preds, 'pred.csv')         # save prediction file to pred.csv
```

### 代码解释

| 代码 | 解释 |
|---|---|
| `test(tt_set, model, device)` | 用训练好的模型对测试集预测。 |
| `preds` | 所有测试样本的预测结果。 |
| `save_pred(preds, 'pred.csv')` | 保存为提交文件。 |
| `writer.writerow(['id', 'tested_positive'])` | 写入 CSV 表头。 |
| `enumerate(preds)` | 同时得到样本编号和预测值。 |

---

## 13. 提升思路

### 13.1 作业提示

| 方向 | 可尝试做法 |
|---|---|
| 特征选择 | 完成 `target_only=True` 的 TODO，只用 40 个州 + 2 个 `tested_positive`。 |
| 模型结构 | 增加层数、宽度，尝试 `Dropout`、`BatchNorm`。 |
| 优化器 | 从 `SGD` 改为 `Adam` 或 `AdamW`。 |
| 学习率 | 调整 `lr`，或加入学习率调度器。 |
| 正则化 | 使用 L2 regularization / weight decay。 |
| 数据处理 | 尝试不同标准化方式，例如 Min-Max normalization。 |

### 13.2 最重要的学习主线

```text
Dataset 负责“准备单条数据”
DataLoader 负责“批量喂数据”
Model 负责“从输入得到预测”
Loss 负责“衡量预测错多少”
Backward 负责“计算梯度”
Optimizer 负责“更新参数”
Dev/Test 负责“评估和预测”
```

> **总结**：这份 notebook 展示了一个完整 PyTorch 回归任务的训练骨架。理解这套流程后，迁移到分类、图像、NLP、推荐系统等任务时，只需要替换数据、模型和损失函数。
