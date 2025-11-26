# SHARE_MLSpring2021_HW2_2 逐行讲解版

面向零基础同学，对 `SHARE_MLSpring2021_HW2_2.ipynb` 中的代码逐行解释，帮助理解 Hessian（海森矩阵）在判断模型处于极小值/鞍点的作用。按 notebook 顺序讲解。

## 学号填写

```python
student_id = 'your_student_id' # fill with your student ID

assert student_id != 'your_student_id', 'Please fill in your student_id before you start.'
```
- `student_id = 'your_student_id'`：请替换为自己的学号字符串。
- `assert ...`：若未替换则抛出错误，提醒必须先填学号（不同学号对应不同数据）。

## 安装 Hessian 计算库

```python
!pip install autograd-lib
```
- 使用 pip 安装 `autograd-lib`（自动求 Hessian 的库）；在 Colab 里运行。

## 导入依赖

```python
import numpy as np
from math import pi
from collections import defaultdict
from autograd_lib import autograd_lib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")
```
- `numpy`：数组计算。
- `pi`：数学常数 π。
- `defaultdict`：便捷的字典初始值。
- `autograd_lib`：用于钩子方式计算 Hessian。
- `torch`/`nn` 等：PyTorch 模型与张量。
- `warnings.filterwarnings("ignore")`：忽略警告输出，保持日志简洁。

## 定义简单回归模型

```python
class MathRegressor(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
```
- `MathRegressor`：拟合单变量函数 `sin(5πx)/(5πx)` 的小型 MLP。
- 输入 1 维，隐藏层 128，ReLU 激活，输出 1 维回归值。
- `forward`：直接调用 `self.regressor`。

## 下载助教准备的检查点

```python
!gdown --id 1ym6G7KKNkbsqSnMmnxdQKHO1JBoF0LPR
```
- 用 gdown 下载 `data.pth`，里面包含不同学号对应的模型与数据。

## 根据学号选择对应数据

```python
# find the key from student_id
import re

key = student_id[-1]
if re.match('[0-9]', key) is not None:
    key = int(key)
else:
    key = ord(key) % 10
```
- 取学号最后一位作为 `key`。
- 若最后一位是数字，直接转 int；否则取字符的 ASCII 码对 10 取模，保证 `key` 为 0~9 之间。

```python
# load checkpoint and data corresponding to the key
model = MathRegressor()
autograd_lib.register(model)

data = torch.load('data.pth')[key]
model.load_state_dict(data['model'])
train, target = data['data']
```
- `model = MathRegressor()`：实例化模型。
- `autograd_lib.register(model)`：让 autograd-lib 跟踪模型各层，方便之后挂钩。
- `data = torch.load('data.pth')[key]`：从下载的字典中取出对应 `key` 的模型与数据。
- `model.load_state_dict(data['model'])`：加载预训练权重。
- `train, target = data['data']`：取出训练输入张量和目标张量（用于 Hessian 计算）。

## 计算梯度范数的函数

```python
# function to compute gradient norm
def compute_gradient_norm(model, criterion, train, target):
    model.train()
    model.zero_grad()
    output = model(train)
    loss = criterion(output, target)
    loss.backward()

    grads = []
    for p in model.regressor.children():
        if isinstance(p, nn.Linear):
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)

    grad_mean = np.mean(grads) # compute mean of gradient norms

    return grad_mean
```
- `model.train()`：启用训练模式。
- `model.zero_grad()`：清空历史梯度。
- `output = model(train)` / `loss = criterion(output, target)`：前向计算 MSE 损失。
- `loss.backward()`：反向传播得到梯度。
- 遍历 `model.regressor` 的子模块，筛选 `nn.Linear` 层。
- `p.weight.grad.norm(2)`：求权重梯度的 L2 范数。
- 收集所有层梯度范数，取均值 `grad_mean` 作为整体梯度强度。

## Hessian 计算相关辅助函数

```python
# helper function to save activations
def save_activations(layer, A, _):
    '''
    A is the input of the layer, we use batch size of 6 here
    layer 1: A has size of (6, 1)
    layer 2: A has size of (6, 128)
    '''
    activations[layer] = A

# helper function to compute Hessian matrix
def compute_hess(layer, _, B):
    '''
    B is the backprop value of the layer
    layer 1: B has size of (6, 128)
    layer 2: B ahs size of (6, 1)
    '''
    A = activations[layer]
    BA = torch.einsum('nl,ni->nli', B, A) # do batch-wise outer product

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA) # do batch-wise outer product, then sum over the batch
```
- `save_activations(layer, A, _)`：前向钩子，保存每层的输入 `A`（激活）。
- `activations[layer] = A`：存储到全局字典。
- `compute_hess(layer, _, B)`：反向钩子，`B` 是反向传播的梯度（对输出）。
- `BA = torch.einsum('nl,ni->nli', B, A)`：对每个样本做外积，得到梯度与输入的组合。
- `hess[layer] += torch.einsum('nli,nkj->likj', BA, BA)`：累加得到近似的完整 Hessian 张量（Gauss-Newton 近似）。

```python
# function to compute the minimum ratio
def compute_minimum_ratio(model, criterion, train, target):
    model.zero_grad()
    # compute Hessian matrix
    # save the gradient of each layer
    with autograd_lib.module_hook(save_activations):
        output = model(train)
        loss = criterion(output, target)

    # compute Hessian according to the gradient value stored in the previous step
    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(output, loss='LeastSquares')

    layer_hess = list(hess.values())
    minimum_ratio = []

    # compute eigenvalues of the Hessian matrix
    for h in layer_hess:
        size = h.shape[0] * h.shape[1]
        h = h.reshape(size, size)
        h_eig = torch.symeig(h).eigenvalues # torch.symeig() returns eigenvalues and eigenvectors of a real symmetric matrix
        num_greater = torch.sum(h_eig > 0).item()
        minimum_ratio.append(num_greater / len(h_eig))

    ratio_mean = np.mean(minimum_ratio) # compute mean of minimum ratio

    return ratio_mean
```
- `model.zero_grad()`：清梯度。
- `with autograd_lib.module_hook(save_activations):`：注册前向钩子捕获输入激活。
- 在钩子环境中前向计算 `output` 和 `loss`。
- `with autograd_lib.module_hook(compute_hess):`：注册反向钩子，在 `autograd_lib.backward_hessian` 时使用。
- `autograd_lib.backward_hessian(output, loss='LeastSquares')`：按最小二乘损失计算 Hessian（使用自动求导库）。
- `layer_hess = list(hess.values())`：取出各层 Hessian。
- 对每层 Hessian：
  - `size = h.shape[0] * h.shape[1]`：行列数展开。
  - `h = h.reshape(size, size)`：重塑为二维矩阵。
  - `h_eig = torch.symeig(h).eigenvalues`：求对称矩阵特征值。
  - `num_greater = torch.sum(h_eig > 0).item()`：统计正特征值数量。
  - `minimum_ratio.append(num_greater / len(h_eig))`：正特征值比例。
- `ratio_mean = np.mean(minimum_ratio)`：对各层比例求平均，作为最终最小比例。
- 返回 `ratio_mean`。

## 计算并打印结果

```python
# the main function to compute gradient norm and minimum ratio
def main(model, train, target):
    criterion = nn.MSELoss()

    gradient_norm = compute_gradient_norm(model, criterion, train, target)
    minimum_ratio = compute_minimum_ratio(model, criterion, train, target)

    print('gradient norm: {}, minimum ratio: {}'.format(gradient_norm, minimum_ratio))
```
- `criterion = nn.MSELoss()`：使用均方误差损失。
- 分别调用梯度范数与最小比例计算函数。
- `print`：输出两个指标。

```python
if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(0)

    # reset compute dictionaries
    activations = defaultdict(int)
    hess = defaultdict(float)

    # compute Hessian
    main(model, train, target)
```
- `torch.manual_seed(0)`：固定随机性（尽管主要操作是确定性的）。
- 初始化全局字典 `activations`、`hess`，用于在钩子中存储激活和 Hessian 累积。
- 调用 `main` 计算并打印 `gradient norm` 与 `minimum ratio`。
- 根据输出与题目给定的判定规则（梯度范数 < 1e-3 且最小比例 > 0.5 判定为“local minima like”，否则结合条件判定鞍点或其他），选择正确答案提交。

> 以上覆盖 notebook 所有代码行，帮助理解 Hessian 计算流程及如何用梯度范数与正特征值比例判断模型状态。
