# SHARE_MLSpring2021_HW2_1 逐行讲解版

面向零基础同学，对 `SHARE_MLSpring2021_HW2_1.ipynb` 中的代码逐行解释，帮助理解语音帧的多分类任务（TIMIT 音素分类）。代码块后紧跟对应解释，顺序与 notebook 一致。

## 下载与解压数据

```python
!gdown --id '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR' --output data.zip
!unzip data.zip
!ls 
```
- `!gdown --id ... --output data.zip`：用 gdown 从 Google Drive 按 ID 下载压缩包到当前目录。
- `!unzip data.zip`：解压出 `timit_11/` 目录，含训练/测试特征与标签。
- `!ls`：列出当前目录，确认解压结果。

## 载入数据

```python
import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))
```
- `import numpy as np`：导入 NumPy 做数组运算。
- `data_root='./timit_11/'`：数据所在目录。
- `np.load(...train_11.npy)`：加载训练特征矩阵，形状 `(1229932, 429)`，每行是 429 维声学特征。
- `np.load(...train_label_11.npy)`：加载训练标签向量。
- `np.load(...test_11.npy)`：加载测试特征。
- `print` 行打印训练/测试数据尺寸，便于确认。

## 定义 Dataset

```python
import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)
```
- `class TIMITDataset(Dataset):`：自定义数据集，继承 PyTorch `Dataset`。
- `self.data = torch.from_numpy(X).float()`：将 NumPy 特征转为 float32 张量。
     NumPy 的 数组（np.ndarray）和 PyTorch 的 张量（torch.Tensor）本质上都是“多维数字容器”。
     它们都支持形状、索引、切片、广播等操作，本质上都是“装数字的数据结构”。
     
- `if y is not None:`：有标签时处理标签。
- `y = y.astype(np.int)`：将标签转为整数类型（旧写法，等价于 int64）。
- `self.label = torch.LongTensor(y)`：转为 PyTorch 长整型张量以用于交叉熵。
  
       
- `else: self.label = None`：测试集无标签。
- `__getitem__`：有标签返回 `(特征, 标签)`，无标签仅返回特征。
- `__len__`：返回样本总数，供 DataLoader 取长度。

## 划分训练/验证集

```python
VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
```
- `VAL_RATIO = 0.2`：20% 数据做验证。
- `percent = int(train.shape[0] * (1 - VAL_RATIO))`：计算训练集样本数。
- `train_x, train_y = train[:percent], train_label[:percent]`：前 80% 做训练。
- `val_x, val_y = train[percent:], train_label[percent:]`：后 20% 做验证。
- `print`：输出划分后尺寸。

## 构建 DataLoader

```python
BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
```
- `BATCH_SIZE = 64`：每批 64 条样本。
- `train_set/val_set = TIMITDataset(...)`：实例化数据集。
- `DataLoader(..., shuffle=True)`：训练集打乱，验证集不打乱；按批输出。

## 释放无用变量节省内存

```python
import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()
```
- `del ...`：删除已用完的大数组释放内存（Colab 容量有限）。
- `gc.collect()`：手动触发垃圾回收。

## 定义模型

```python
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39) 

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```
- `class Classifier(nn.Module):`：多层全连接网络，用于 39 类音素分类。
- `layer1/2/3`：三层线性映射，将 429 维输入压缩到 1024→512→128。
- `self.out = nn.Linear(128, 39)`：输出 39 维 logits，对应类别。
- `self.act_fn = nn.Sigmoid()`：使用 Sigmoid 作为隐藏层激活（可尝试 ReLU/LeakyReLU 改进）。
- `forward`：依次线性→激活，最后输出 logits，不做 softmax，交叉熵会内部处理。

## 训练准备工具

```python
#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'
```
- `get_device`：检测可用 GPU，返回设备字符串。

```python
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```
- `same_seeds(seed)`：固定随机性，保证复现；包括 CPU/GPU 种子，关闭 CuDNN benchmark 以确定性运行。

## 配置训练参数

```python
# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 20               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
- `same_seeds(0)`：用种子 0 固定实验。
- `device = get_device()` / `print`：确认运行设备。
- `num_epoch = 20`：训练 20 轮。
- `learning_rate = 0.0001`：Adam 学习率。
- `model_path = './model.ckpt'`：保存最佳权重的路径。
- `model = Classifier().to(device)`：实例化模型并移到设备。
- `criterion = nn.CrossEntropyLoss()`：多分类交叉熵损失，需要 logits 和长整型标签。
- `optimizer = torch.optim.Adam(...)`：Adam 优化器。

## 训练循环

```python
# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
```
- `best_acc = 0.0`：记录最佳验证正确数。
- `for epoch in range(num_epoch):`：遍历各训练轮。
- 初始化当轮的累计指标 `train_acc/train_loss/val_acc/val_loss`。
- `model.train()`：训练模式，启用 dropout/BN 等。
- `for i, data in enumerate(train_loader):`：遍历训练批次。
- `inputs, labels = data`：解包特征和标签。
- `inputs, labels = inputs.to(device), labels.to(device)`：移到设备。
- `optimizer.zero_grad()`：清零梯度。
- `outputs = model(inputs)`：前向计算 logits。
- `batch_loss = criterion(outputs, labels)`：交叉熵损失。
- `_, train_pred = torch.max(outputs, 1)`：取每行最大值下标作为预测类别。
- `batch_loss.backward()`：反向传播。
- `optimizer.step()`：更新参数。
- `train_acc += ...`：累加预测正确样本数。
- `train_loss += batch_loss.item()`：累加批次损失。
- 验证部分 `if len(val_set) > 0:`：存在验证集则评估。
- `model.eval()`：评估模式，关闭 dropout/BN 统计更新。
- `with torch.no_grad():`：禁用梯度，提高推理效率。
- 循环验证批次，计算 `batch_loss` 和 `val_pred`，累加准确数与损失。
- `print('[...].format(...)`：按轮打印训练/验证准确率与平均损失。
- `if val_acc > best_acc:`：验证正确数更高则保存权重。
- `torch.save(model.state_dict(), model_path)`：只保存模型参数。
- `else:` 分支：若无验证集，仅打印训练指标。
- 末尾 `if len(val_set) == 0:`：无验证集时保存最后一轮模型。

## 测试与生成提交文件

```python
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))
```
- `test_set = TIMITDataset(test, None)`：测试集无标签。
- `DataLoader(... shuffle=False)`：保持原顺序输出。
- 重新实例化模型并移到设备。
- `model.load_state_dict(torch.load(model_path))`：加载之前验证最佳的权重。

```python
predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)
```
- `predict = []`：收集预测类别。
- `model.eval()` / `with torch.no_grad():`：评估模式、关闭梯度。
- 遍历测试批次，前向得到 `outputs`，取最大值索引为类别 `test_pred`。
- 将预测搬到 CPU、转 NumPy，追加到列表。

```python
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
```
- 打开/创建 `prediction.csv` 写入。
- 首行写表头 `Id,Class`。
- 枚举预测列表，将索引作为样本 ID，与预测类别写入 CSV。
- 生成的文件可提交竞赛平台评分。

> 以上覆盖 notebook 所有代码行，便于理解数据处理、模型构建、训练验证与测试输出的完整流程。
