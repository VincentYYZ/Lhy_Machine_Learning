# ML2021Spring_HW1 逐行讲解版

面向零基础同学，对 `ML2021Spring_HW1.ipynb` 中的每行代码做中文解释，并穿插相关的机器学习概念。代码块后紧跟对应解释，顺序与 notebook 一致。

## 下载数据

```python
tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'   # path to testing data

!gdown --id '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF' --output covid.train.csv
!gdown --id '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O' --output covid.test.csv
```
- `tr_path = 'covid.train.csv'`：设定训练集文件名；训练集包含特征和真实标签。
- `tt_path = 'covid.test.csv'`：设定测试集文件名；测试集只有特征没有标签，用于提交预测。
- `!gdown ... --output covid.train.csv`：在 Colab/命令行使用 `gdown` 根据 Google Drive ID 下载训练集到当前目录。
- `!gdown ... --output covid.test.csv`：下载测试集。

## 导入包并设置随机种子

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
- `import torch`：导入 PyTorch 主库，用于张量计算与训练。
- `import torch.nn as nn`：导入神经网络层与损失函数模块。
- `from torch.utils.data import Dataset, DataLoader`：数据集与批量加载工具。
- `import numpy as np`：NumPy，用于数组处理和数值计算。
- `import csv`：读取 CSV 文件的标准库。
- `import os`：文件和路径操作。
- `import matplotlib.pyplot as plt` / `from matplotlib.pyplot import figure`：绘图函数，绘制学习曲线与预测散点图。
- `myseed = 42069`：固定随机种子，保证实验可复现。
- `torch.backends.cudnn.deterministic = True`：让 CuDNN 的算法确定性，减少随机性。
- `torch.backends.cudnn.benchmark = False`：关闭算法自动搜索，配合上行确保结果一致。
- `np.random.seed(myseed)`：固定 NumPy 随机性。
- `torch.manual_seed(myseed)`：固定 CPU 端 PyTorch 随机性。
- `if torch.cuda.is_available(): torch.cuda.manual_seed_all(myseed)`：若有 GPU，则固定 GPU 端随机性。

## 常用工具函数

```python
def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'
```
- `def get_device():`：定义函数检测设备。
- `return 'cuda' if torch.cuda.is_available() else 'cpu'`：若有 GPU 返回 `'cuda'`，否则返回 `'cpu'`。

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
- `total_steps = len(loss_record['train'])`：训练记录的步数。
- `x_1 = range(total_steps)`：训练曲线的横轴。
- `x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]`：把验证集记录均匀映射到横轴。
- `figure(figsize=(6, 4))`：设定图尺寸。
- `plt.plot(... train ...)`：红色曲线画训练损失。
- `plt.plot(... dev ...)`：青色曲线画验证损失。
- `plt.ylim(0.0, 5.)`：Y 轴范围 0~5，方便观察。
- `plt.xlabel('Training steps')` / `plt.ylabel('MSE loss')`：坐标轴标签。
- `plt.title(...)`：图标题。
- `plt.legend()`：显示图例。
- `plt.show()`：展示图形。

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
- `def plot_pred(...)`：绘制预测值与真实值散点，检查模型拟合。
- `if preds is None or targets is None:`：若未传入预计算结果，则现场计算。
- `model.eval()`：切换评估模式，禁用 dropout/BN 等训练行为。
- `preds, targets = [], []`：准备收集预测与标签。
- `for x, y in dv_set:`：遍历验证集批次。
- `x, y = x.to(device), y.to(device)`：把数据移到 CPU/GPU。
- `with torch.no_grad(): pred = model(x)`：关闭梯度，仅做前向推理。
- `preds.append(pred.detach().cpu())`：取出预测张量、搬到 CPU 并累加。
- `targets.append(y.detach().cpu())`：同理收集真实标签。
- `preds = torch.cat(...).numpy()` / `targets = ...`：拼接所有批次并转为 NumPy，便于绘图。
- `figure(figsize=(5, 5))`：图尺寸。
- `plt.scatter(targets, preds, c='r', alpha=0.5)`：散点图，红色、半透明。
- `plt.plot([-0.2, lim], [-0.2, lim], c='b')`：画对角线，理想预测应落在此线附近。
- `plt.xlim/ylim(-0.2, lim)`：设置坐标范围。
- `plt.xlabel/plt.ylabel(...)`：轴标签。
- `plt.title('Ground Truth v.s. Prediction')`：标题。
- `plt.show()`：显示图。

## 数据预处理与 Dataset

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
- `class COVID19Dataset(Dataset):`：继承 `torch.utils.data.Dataset`，定义自定数据集。
- `__init__(path, mode='train', target_only=False)`：初始化参数，`mode` 控制训练/验证/测试，`target_only` 是否只选部分特征。
- `self.mode = mode`：保存模式。
- `with open(path, 'r') as fp:`：打开 CSV。
- `data = list(csv.reader(fp))`：逐行读取为列表。
- `data = np.array(data[1:])[:, 1:].astype(float)`：跳过表头行、跳过第一列 ID，仅保留数值特征并转为浮点。
- `if not target_only: feats = list(range(93))`：默认使用 93 个输入特征。
- `else: ... pass`：TODO：若开启只用目标相关特征，这里需指定 40 个州 + 2 个 `tested_positive` 特征（索引 57 和 75）。
- `if mode == 'test':`：测试模式不含标签。
- `data = data[:, feats]`：按挑选的特征列截取。
- `self.data = torch.FloatTensor(data)`：测试集保存为浮点张量。
- `else:`：训练/验证模式包含标签。
- `target = data[:, -1]`：最后一列是标签（未来第 4 天确诊数）。
- `data = data[:, feats]`：特征截取。
- `if mode == 'train': indices = [i for i ... if i % 10 != 0]`：按 9:1 划分，取 90% 作为训练。
- `elif mode == 'dev': indices = [i for i ... if i % 10 == 0]`：取每 10 个样本中的第 0 个作为验证。
- `self.data = torch.FloatTensor(data[indices])`：选定样本转张量。
- `self.target = torch.FloatTensor(target[indices])`：选定标签转张量。
- `self.data[:, 40:] = (self.data[:, 40:] - mean) / std`：对第 40 列及之后的时间序列特征做标准化（减均值除标准差），让各特征尺度相近；提高收敛速度，防止梯度不稳定。
- `self.dim = self.data.shape[1]`：记录特征维度。
- `print('Finished reading ...')`：输出读入信息，便于确认维度和样本数。
- `__getitem__`：按索引返回一条数据；训练/验证返回 `(特征, 标签)`，测试仅返回特征。
- `__len__`：返回数据集大小，供 `DataLoader` 使用。

## DataLoader

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
- `prep_dataloader(...)`：封装数据集创建和批量加载。
- `dataset = COVID19Dataset(...)`：实例化自定义数据集。
- `DataLoader(... shuffle=(mode=='train') ...)`：训练集打乱样本，验证/测试不打乱；`drop_last=False` 保留最后不足一批的数据；`num_workers` 控制并行加载；`pin_memory=True` 便于 GPU 加速。
- `return dataloader`：返回可迭代的批数据对象。

## 定义神经网络

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
- `class NeuralNet(nn.Module):`：定义继承自 `nn.Module` 的模型。
- `__init__(input_dim)`：传入特征维度。
- `self.net = nn.Sequential(...)`：顺序容器搭建两层全连接网络；第一层把输入映射到 64 维，`nn.ReLU()` 激活引入非线性；第二层输出 1 维回归值。
- `self.criterion = nn.MSELoss(reduction='mean')`：均方误差作为回归损失。
- `forward`：前向传播，把输入喂给 `self.net`，`squeeze(1)` 去掉多余的维度，得到 `(batch,)` 张量。
- `cal_loss`：计算损失，目前直接用 MSE；注释提示可以加入 L2 正则（权重衰减）提升泛化。

## 训练、验证与测试函数

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
- `n_epochs = config['n_epochs']`：最大训练轮数。
- `optimizer = getattr(torch.optim, config['optimizer'])(...)`：根据配置创建优化器，这里是 `SGD`。
- `min_mse = 1000.`：记录当前最好验证集损失，初值很大。
- `loss_record = {'train': [], 'dev': []}`：存训练/验证损失曲线。
- `early_stop_cnt = 0` / `epoch = 0`：早停计数与当前轮数。
- `while epoch < n_epochs:`：主训练循环。
- `model.train()`：启用训练模式。
- `for x, y in tr_set:`：遍历训练批次。
- `optimizer.zero_grad()`：清空上一批梯度。
- `x, y = x.to(device), y.to(device)`：数据放到 GPU/CPU。
- `pred = model(x)`：前向得到预测。
- `mse_loss = model.cal_loss(pred, y)`：计算均方误差。
- `mse_loss.backward()`：反向传播计算梯度。
- `optimizer.step()`：按梯度更新参数。
- `loss_record['train'].append(...)`：记录当前批次训练损失。
- `dev_mse = dev(dv_set, model, device)`：每轮结束在验证集评估。
- `if dev_mse < min_mse:`：若验证损失提升则保存模型。
- `torch.save(model.state_dict(), config['save_path'])`：只保存权重参数。
- `early_stop_cnt = 0` / `else: early_stop_cnt += 1`：验证未提升则累加早停计数。
- `epoch += 1`：进入下一轮。
- `loss_record['dev'].append(dev_mse)`：记录验证损失。
- `if early_stop_cnt > config['early_stop']:`：连续若干轮未提升就提前停止，防止过拟合或浪费计算。
- `return min_mse, loss_record`：返回最优验证损失与曲线记录。

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
- `model.eval()`：评估模式，关闭 dropout/BN 统计更新。
- `total_loss = 0`：累计损失。
- `for x, y in dv_set:`：遍历验证批次。
- `x, y = x.to(device), y.to(device)`：移到设备。
- `with torch.no_grad(): pred = model(x)`：无梯度前向。
- `mse_loss = model.cal_loss(pred, y)`：批次 MSE。
- `total_loss += mse_loss ... * len(x)`：按样本数加权累加总损失。
- `total_loss = total_loss / len(dv_set.dataset)`：除以总样本数得到平均损失。
- `return total_loss`：返回验证集平均 MSE。

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
- `model.eval()`：评估模式。
- `preds = []`：收集预测。
- `for x in tt_set:`：遍历测试集批次。
- `x = x.to(device)`：移到设备。
- `with torch.no_grad(): pred = model(x)`：无梯度前向。
- `preds.append(pred.detach().cpu())`：取出预测放入列表。
- `preds = torch.cat(preds, dim=0).numpy()`：拼接成完整预测并转 NumPy。
- `return preds`：返回预测数组。

## 超参数配置

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
- `device = get_device()`：自动选择 CPU/GPU。
- `os.makedirs('models', exist_ok=True)`：创建保存模型的目录，已存在则跳过。
- `target_only = False`：是否只用部分特征；若设为 `True` 需补全 Dataset 中 TODO。
- `config`：训练用超参数字典。
  - `'n_epochs': 3000`：最大训练轮数上限。
  - `'batch_size': 270`：批大小，影响梯度估计稳定性与显存占用。
  - `'optimizer': 'SGD'`：使用随机梯度下降。
  - `'optim_hparas': {'lr': 0.001, 'momentum': 0.9}`：学习率和动量；动量可加速收敛。
  - `'early_stop': 200`：若验证集 200 轮未提升则早停。
  - `'save_path': 'models/model.pth'`：最佳模型权重保存路径。

## 加载数据与模型

```python
tr_set = prep_dataloader(tr_path, 'train', config['batch_size'], target_only=target_only)
dv_set = prep_dataloader(tr_path, 'dev', config['batch_size'], target_only=target_only)
tt_set = prep_dataloader(tt_path, 'test', config['batch_size'], target_only=target_only)
```
- `prep_dataloader(... 'train' ...)`：构造训练集 DataLoader。
- `prep_dataloader(... 'dev' ...)`：构造验证集 DataLoader。
- `prep_dataloader(... 'test' ...)`：构造测试集 DataLoader。
- 终端输出类似“Finished reading ... dim = 93”，确认样本数与维度。

```python
model = NeuralNet(tr_set.dataset.dim).to(device)  # Construct model and move to device
```
- `NeuralNet(tr_set.dataset.dim)`：用数据集的特征维度实例化模型。
- `.to(device)`：把模型参数移到 CPU 或 GPU。

## 开始训练

```python
model_loss, model_loss_record = train(tr_set, dv_set, model, config, device)
```
- 调用训练函数，返回最优验证损失和损失曲线。
- 训练过程中会在验证集提升时打印“Saving model (epoch = ..., loss = ...)”，并保存权重到 `config['save_path']`。
- 日志显示损失逐步下降，说明模型在学习；早停触发时结束训练。

```python
plot_learning_curve(model_loss_record, title='deep model')
```
- 使用前面工具函数绘制训练/验证损失曲线，观察是否过拟合或欠拟合。

```python
del model
model = NeuralNet(tr_set.dataset.dim).to(device)
ckpt = torch.load(config['save_path'], map_location='cpu')  # Load your best model
model.load_state_dict(ckpt)
plot_pred(dv_set, model, device)  # Show prediction on the validation set
```
- `del model`：释放旧模型与显存。
- 重新实例化同结构模型并移到设备。
- `torch.load(... map_location='cpu')`：加载保存的最佳权重；映射到 CPU 方便通用。
- `model.load_state_dict(ckpt)`：把权重装回模型。
- `plot_pred(dv_set, model, device)`：绘制验证集预测 vs 真实散点，靠近对角线表示拟合良好。

## 测试并保存提交文件

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
- `save_pred(preds, file)`：把预测结果写入 CSV，含两列：`id`（样本序号）和 `tested_positive`（预测值）。
- `preds = test(tt_set, model, device)`：用训练好的模型在测试集上推理。
- `save_pred(preds, 'pred.csv')`：保存为 `pred.csv`，用于比赛提交或评分。

## 提示与提升思路

- **特征选择**：完成 Dataset 中的 TODO，尝试仅用 40 州指标 + 2 个 `tested_positive` 特征，可能更稳。
- **网络结构**：增加层数/宽度、加入 `Dropout` 或 `BatchNorm`，尝试不同激活函数提升表现。
- **训练策略**：改用 `Adam` 优化器、更高/自适应学习率、调整批大小；尝试学习率衰减调度。
- **正则化**：在 `cal_loss` 中加入 L2（权重衰减）减少过拟合。
- **数据标准化**：可以尝试只标准化部分特征或改成 Min-Max 归一化，对模型收敛有影响。

> 以上解释覆盖 notebook 中全部代码行，便于逐行理解深度学习回归任务的实现流程。祝学习顺利！
