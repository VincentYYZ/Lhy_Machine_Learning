# hw6_GAN 逐行讲解版

面向零基础同学，对 `hw6_GAN.ipynb` 代码按顺序逐段解释，帮助理解 DCGAN 训练流程（动漫头像生成）。代码块后紧跟说明。

## 环境与依赖

```python
workspace_dir = '.'
!pip install -q qqdm
```
- 设置工作目录。
- 安装 `qqdm` 进度条（notebook 友好版）。

## 下载数据

```python
!gdown --id 1IGrTr308mGAaCKotpkkm8wTKlWs9Jq-p --output "{workspace_dir}/crypko_data.zip"
# 其他学号尾号对应的 gdown 链接在注释中
```
- 按学号尾号选用对应的 Google Drive ID 下载 `crypko_data.zip`。

### 解压

```python
!unzip -q "{workspace_dir}/crypko_data.zip" -d "{workspace_dir}/"
```
- 解压到工作目录，得到 `faces/`，包含形如 `1.jpg` 的人脸图。

## 随机种子

```python
import random, torch, numpy as np

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2021)
```
- 固定 Python、NumPy、PyTorch（含 GPU）随机性，关闭 CuDNN 自适应以确保结果可复现。

## 导入包

```python
import os, glob
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from qqdm.notebook import qqdm
```
- 文件/路径、PyTorch 模块、torchvision 数据增强/可视化、自动求导 Variable（旧写法，与 tensor 等价）、自定义数据集/DataLoader、绘图、进度条。

## 数据集

```python
class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)   # 读取为张量 [C,H,W], 0~255
        img = self.transform(img)                # 变换：resize+归一化
        return img

    def __len__(self):
        return self.num_samples
```
- 自定义数据集：存储文件名列表，按索引读取图片并应用 transform，返回张量。

```python
def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 将[0,1]线性映射到[-1,1]
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset
```
- 收集 `faces/` 下图片路径；transform：转 PIL → 缩放 64x64 → 转张量（0~1）→ 归一化到 [-1,1]；返回数据集。

### 可视化

```python
dataset = get_dataset(os.path.join(workspace_dir, 'faces'))
images = [dataset[i] for i in range(16)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.imshow(grid_img.permute(1, 2, 0)); plt.show()

images = [(dataset[i]+1)/2 for i in range(16)]  # 还原到[0,1]再显示
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.imshow(grid_img.permute(1, 2, 0)); plt.show()
```
- 先直接显示（需要 clip）；再把 [-1,1] 映射回 [0,1] 以正常显示。

## 模型：DCGAN

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
```
- 权重初始化：卷积权重 N(0,0.02)，BN 权重 N(1,0.02)，偏置 0。

```python
class Generator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),   # 4x4 -> 8x8
            dconv_bn_relu(dim * 4, dim * 2),   # 8x8 -> 16x16
            dconv_bn_relu(dim * 2, dim),       # 16x16 -> 32x32
            nn.ConvTranspose2d(dim, 3, 5, 2, padding=2, output_padding=1),  # 32x32 -> 64x64, 3通道
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)   # reshape 为 feature map
        y = self.l2_5(y)
        return y
```
- 生成器：输入噪声 (N, z_dim)，线性映射到 4x4x512，再用转置卷积逐步上采样到 64x64，Tanh 输出 [-1,1]。

```python
class Discriminator(nn.Module):
    def __init__(self, in_dim, dim=64):
        super().__init__()
        def conv_bn_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU(0.2),
            )
        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), 
            nn.LeakyReLU(0.2),
            conv_bn_lrelu(dim, dim * 2),
            conv_bn_lrelu(dim * 2, dim * 4),
            conv_bn_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, 1, 4),
            nn.Sigmoid(),   # Medium 提示：WGAN 可去掉 Sigmoid
        )
        self.apply(weights_init)
        
    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y
```
- 判别器：卷积下采样 64→32→16→8→4，最后 1x1 输出真实概率标量；LeakyReLU 激活。

## 训练初始化

```python
batch_size = 64
z_dim = 100
z_sample = Variable(torch.randn(100, z_dim)).cuda()  # 用于定期可视化
lr = 1e-4
n_epoch = 1      # Medium 建议 50
n_critic = 1     # WGAN 时可用 5
# clip_value = 0.01  # WGAN 权重裁剪

log_dir = './logs'; ckpt_dir = './checkpoints'; os.makedirs(..., exist_ok=True)

G = Generator(in_dim=z_dim).cuda(); D = Discriminator(3).cuda()
G.train(); D.train()
criterion = nn.BCELoss()
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# WGAN 选项：RMSprop

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
```
- 设定超参数、准备日志/模型目录、实例化模型、损失（标准 GAN 用 BCE）、优化器（Adam；WGAN 可改 RMSprop）、DataLoader。

## 训练循环

```python
steps = 0
for e, epoch in enumerate(range(n_epoch)):
    progress_bar = qqdm(dataloader)
    for i, data in enumerate(progress_bar):
        imgs = data.cuda()
        bs = imgs.size(0)

        # --- 训练 D ---
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()      # 真实图片
        f_imgs = G(z)                       # 生成图片
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())
        r_loss = criterion(r_logit, r_label)
        f_loss = criterion(f_logit, f_label)
        loss_D = (r_loss + f_loss) / 2      # GAN 判别器损失
        # WGAN: loss_D = -E[D(real)] + E[D(fake)]
        D.zero_grad()
        loss_D.backward()
        opt_D.step()
        # WGAN: 可对 D 权重裁剪

        # --- 训练 G ---
        if steps % n_critic == 0:
            z = Variable(torch.randn(bs, z_dim)).cuda()
            f_imgs = G(z)
            f_logit = D(f_imgs)
            loss_G = criterion(f_logit, r_label)   # 让假图判为真
            # WGAN: loss_G = -E[D(fake)]
            G.zero_grad()
            loss_G.backward()
            opt_G.step()

        steps += 1
        progress_bar.set_infos({'Loss_D': round(loss_D.item(),4),
                                'Loss_G': round(loss_G.item(),4),
                                'Epoch': e+1, 'Step': steps})

    # 每轮结束：保存样例图、可视化
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    filename = os.path.join(log_dir, f'Epoch_{epoch+1:03d}.jpg')
    torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
    grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
    plt.imshow(grid_img.permute(1, 2, 0)); plt.show()
    G.train()

    if (e+1) % 5 == 0 or e == 0:
        torch.save(G.state_dict(), os.path.join(ckpt_dir, 'G.pth'))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, 'D.pth'))
```
- 判别器：真实/生成各自 BCE，求均值，反传更新；可选 WGAN 损失与裁剪。
- 生成器：每 `n_critic` 步更新一次，目标是判别器输出真。
- 进度条显示当前损失；每轮保存样本和 checkpoint。

## 推理生成

```python
G = Generator(z_dim)
G.load_state_dict(torch.load(os.path.join(ckpt_dir, 'G.pth')))
G.eval(); G.cuda()
```
- 加载已训练的生成器权重，切换到 eval 模式。

```python
n_output = 1000
z_sample = Variable(torch.randn(n_output, z_dim)).cuda()
imgs_sample = (G(z_sample).data + 1) / 2.0  # 还原到[0,1]
log_dir = './logs'; filename = os.path.join(log_dir, 'result.jpg')
torchvision.utils.save_image(imgs_sample, filename, nrow=10)
grid_img = torchvision.utils.make_grid(imgs_sample[:32].cpu(), nrow=10)
plt.imshow(grid_img.permute(1, 2, 0)); plt.show()
```
- 生成 1000 张图片，保存网格，显示前 32 张。

### 打包输出

```python
os.makedirs('output', exist_ok=True)
for i in range(1000):
    torchvision.utils.save_image(imgs_sample[i], f'output/{i+1}.jpg')
%cd output
!tar -zcf ../images.tgz *.jpg
%cd ..
```
- 将单张图片保存到 `output/`，打包为 `images.tgz` 便于提交。

> 以上覆盖 notebook 全部代码行，逐段解释数据处理、DCGAN 结构、训练循环、WGAN 选项、推理与导出步骤。可尝试提高 epoch、启用 WGAN（去 Sigmoid、换损失/优化器、权重裁剪）、或改进模型（如加入残差、谱归一化）以提升生成质量。祝实验顺利！
