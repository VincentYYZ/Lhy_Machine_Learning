# hw12_reinforcement_learning_english_version 逐行讲解版

面向零基础同学，对 `hw12_reinforcement_learning_english_version.ipynb` 按顺序逐段解释，帮助理解在 Gym LunarLander-v2 上实现 REINFORCE + baseline 的强化学习流程。代码块后附说明。

## 环境与依赖

```python
!apt update
!apt install python-opengl xvfb -y
!pip install gym[box2d]==0.18.3 pyvirtualdisplay tqdm numpy==1.19.5 torch==1.8.1
```
- 安装 OpenGL/Xvfb（渲染需要），Gym box2d 环境，虚拟显示，指定版本的 numpy/torch/tqdm。

```python
from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()
```
- 启动虚拟显示以支持渲染（Colab 无物理屏幕）。

```python
import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
```
- 导入常用库、PyTorch 组件、Categorical 分布用于采样动作。

## 固定随机种子（不要改）

```python
seed = 543
def fix(env, seed):
    env.seed(seed); env.action_space.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```
- 为环境和各库设定随机种子确保可复现。作业要求不要修改种子。

## 创建环境

```python
import gym, random
env = gym.make('LunarLander-v2')
fix(env, seed)
```
- 构建 LunarLander-v2 环境，包含 8 维状态和 4 离散动作。

## 状态与动作空间

```python
print(env.observation_space)  # Box(8,)
print(env.action_space)       # Discrete(4)
```
- 状态：8 维连续向量；动作：4 种（0 无推力，1 左，2 主引擎下，3 右）。

## 网络定义

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=8, action_dim=4, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
    def forward(self, x): return self.net(x)
```
- 策略网络：两层 MLP 输出动作概率。

```python
class ValueNetwork(nn.Module):
    def __init__(self, state_dim=8, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)
```
- 价值网络：估计状态价值 V(s)，用于 baseline 减小方差。

## 采样与回合运行

```python
def select_action(policy_net, state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy_net(state)
    m = Categorical(probs)
    action = m.sample()
    return action.item(), m.log_prob(action)
```
- 将状态转 tensor，前向得到动作分布，采样动作并返回 log_prob。

```python
def run_episode(env, policy_net):
    states, actions_log_prob, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
        action, log_prob = select_action(policy_net, state)
        next_state, reward, done, _ = env.step(action)
        states.append(state); actions_log_prob.append(log_prob); rewards.append(reward)
        state = next_state
    return states, actions_log_prob, rewards
```
- 运行一整个回合，收集状态、动作 log_prob、奖励。

## 折扣回报与优势

```python
def compute_returns(rewards, gamma=0.99):
    R = 0; returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)
```
- 从末尾向前累计折扣回报。

优势计算使用 baseline：

```python
returns = compute_returns(rewards, gamma)
states_tensor = torch.tensor(states, dtype=torch.float32)
values = value_net(states_tensor).detach()
advantages = returns - values
```
- 优势 = 回报 - 估计价值。

## 损失函数

- 策略损失（REINFORCE with baseline）：

```python
policy_loss = []
for log_prob, advantage in zip(actions_log_prob, advantages):
    policy_loss.append(-log_prob * advantage)
policy_loss = torch.stack(policy_loss).sum()
```

- 价值损失（MSE）：

```python
value_loss = F.mse_loss(value_net(states_tensor), returns)
```

## 训练循环

```python
policy_net = PolicyNetwork().to(device)
value_net = ValueNetwork().to(device)
optimizer_policy = optim.Adam(policy_net.parameters(), lr=1e-3)
optimizer_value = optim.Adam(value_net.parameters(), lr=1e-3)

num_epochs = 500
for epoch in range(num_epochs):
    states, actions_log_prob, rewards = run_episode(env, policy_net)
    returns = compute_returns(rewards, gamma=0.99)
    states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
    actions_log_prob = torch.stack(actions_log_prob).to(device)
    returns = returns.to(device)
    values = value_net(states_tensor)
    advantages = returns - values.detach()

    policy_loss = -(actions_log_prob * advantages).sum()
    value_loss = F.mse_loss(values, returns)

    optimizer_policy.zero_grad(); optimizer_value.zero_grad()
    policy_loss.backward(); value_loss.backward()
    optimizer_policy.step(); optimizer_value.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Return: {returns[0]:.2f}, Policy loss: {policy_loss.item():.2f}, Value loss: {value_loss.item():.2f}')
```
- 每回合：采样、计算回报/优势，更新策略和价值网络。
- 打印训练进度（回报/损失）。

## 渲染与评估（可选）

```python
state = env.reset()
for t in range(500):
    action, _ = select_action(policy_net, state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done: break
env.close()
```
- 运行训练后的策略并渲染（需虚拟显示），可录制 gif/视频。

## 关键点与改进
- 不要改 seed 以与评测对齐。
- 超参可调整：学习率、gamma、隐藏层大小、baseline 权重、梯度裁剪等。
- 可加入回报标准化、优势标准化以稳定训练。
- 若需更稳定，可使用多回合采样再批量更新，或采用 GAE/actor-critic 变体。

> 以上逐行解释 notebook 主要代码，涵盖环境搭建、策略/价值网络、回合采样、回报与优势计算、REINFORCE+baseline 训练流程。祝强化学习实验顺利！***
