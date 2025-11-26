# hw9_xai 逐行讲解版

面向零基础同学，对 `hw9_xai.ipynb` 的代码逐段解释，涵盖 CNN 可视化（LIME、Saliency/SmoothGrad、过滤器可视化、集成梯度）和 BERT 可视化提示。按 notebook 顺序讲解。

## Colab 环境与数据准备

```python
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
import os
os.chdir('gdrive/My Drive/MLHW_XAI')
!ls
```
- 挂载 Google Drive，进入作业目录，列出文件。

### 下载数据与预训练模型

```python
!gdown --id '1cYBWwYab3djiaYuOU6CxkYHQyUYws4Ce' --output food.zip
!unzip food.zip
!gdown --id '1CShZHsO8oAZwxQkMe7jRtEgSNb2w_OZu' --output checkpoint.pth
```
- 下载并解压 food-11 数据集；下载已训练好的 checkpoint。

### 安装 LIME

```python
!pip install lime==0.1.1.37
```
- 安装 LIME 库用于可解释性。

## 导入包

```python
import os, sys, argparse, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from skimage.segmentation import slic
from lime import lime_image
from pdb import set_trace
from torch.autograd import Variable
```
- 常规科学计算/深度学习库，图像处理，LIME，SLIC 分割（用于 LIME 超像素）。

## 参数

```python
args = {'ckptpath': './checkpoint.pth', 'dataset_dir': './food/'}
args = argparse.Namespace(**args)
```
- 指定模型权重与数据目录。

## 模型定义与加载

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        def building_block(indim, outdim): ...
        def stack_blocks(indim, outdim, block_num): ...
        cnn_list = []
        cnn_list += stack_blocks(3, 128, 3)
        cnn_list += stack_blocks(128, 128, 3)
        cnn_list += stack_blocks(128, 256, 3)
        cnn_list += stack_blocks(256, 512, 1)
        cnn_list += stack_blocks(512, 512, 1)
        self.cnn = nn.Sequential(*cnn_list)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(p=0.3),
            nn.Linear(1024, 11),
        )
    def forward(self, x):
        out = self.cnn(x)
        out = out.reshape(out.size()[0], -1)
        return self.fc(out)
```
- 多层卷积+BN+ReLU+池化的分类模型，输出 11 类。

```python
model = Classifier().cuda()
checkpoint = torch.load(args.ckptpath)
model.load_state_dict(checkpoint['model_state_dict'])
```
- 加载预训练权重。

## 数据集定义与加载

```python
class FoodDataset(Dataset):
    def __init__(self, paths, labels, mode):
        trainTransform = transforms.Compose([... Resize(128), RandomFlip/Rotation, ToTensor ...])
        evalTransform = transforms.Compose([Resize(128), ToTensor()])
        self.transform = trainTransform if mode=='train' else evalTransform
    def __len__(self): return len(self.paths)
    def __getitem__(self, index):
        X = Image.open(self.paths[index]); X = self.transform(X); Y = self.labels[index]; return X, Y
    def getbatch(self, indices):  # 便于可视化
        ...
```
- `get_paths_labels` 解析文件名中的标签并排序；`train_set = FoodDataset(..., mode='eval')` 生成数据集。

### 可视化样本

```python
img_indices = [0..9]
images, labels = train_set.getbatch(img_indices)
plt.imshow(...)  # 展示 10 张样本
```
- 取若干样本用于后续解释。

## LIME 可解释性

```python
def predict(input):
    model.eval()
    input = torch.FloatTensor(input).permute(0,3,1,2)
    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    return slic(input, n_segments=200, compactness=1, sigma=1)
```
- LIME 需要预测函数（numpy 输入）和分割函数（超像素）。

```python
for image, label in zip(images.permute(0,2,3,1).numpy(), labels):
    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(
        image=image.astype(np.double),
        classifier_fn=predict,
        segmentation_fn=segmentation)
    lime_img, mask = explaination.get_image_and_mask(
        label=label.item(), positive_only=False, hide_rest=False,
        num_features=11, min_weight=0.05)
    plt.imshow(lime_img)
```
- 生成并展示 LIME 解释图。

## Saliency Map

```python
def normalize(image): return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
    model.eval(); x = x.cuda(); x.requires_grad_()
    y_pred = model(x); loss = CrossEntropy(y_pred, y.cuda()); loss.backward()
    saliencies, _ = torch.max(x.grad.data.abs().detach().cpu(), dim=1)
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies
```
- 对输入求梯度，取通道最大绝对值作为显著图，归一化。

```python
saliencies = compute_saliency_maps(images, labels, model)
plt.imshow(images) / plt.imshow(saliencies, cmap=hot)
```
- 显示原图与显著图。

## SmoothGrad

```python
def smooth_grad(x, y, model, epoch, param_sigma_multiplier):
    sigma = param_sigma_multiplier / (torch.max(x) - torch.min(x)).item()
    smooth = np.zeros(x.cuda().unsqueeze(0).size())
    for i in range(epoch):
        noise = Variable(x.data.new(x.size()).normal_(0, sigma**2))
        x_mod = (x+noise).unsqueeze(0).cuda(); x_mod.requires_grad_()
        y_pred = model(x_mod); loss = CrossEntropy(y_pred, y.cuda().unsqueeze(0)); loss.backward()
        smooth += x_mod.grad.abs().detach().cpu().data.numpy()
    smooth = normalize(smooth / epoch)
    return smooth
```
- 给输入加噪，累积梯度，平均后归一化，得到更平滑的显著图。

```python
smooth = [smooth_grad(i,l,model,500,0.4) for i,l in zip(images,labels)]
```
- 生成并展示 SmoothGrad。

## Filter activation/visualization（使用 hook）

```python
layer_activations = None
def filter_explanation(x, model, cnnid, filterid, iteration=100, lr=1):
    def hook(model, input, output):
        global layer_activations
        layer_activations = output
    hook_handle = model.cnn[cnnid].register_forward_hook(hook)
    model(x.cuda())  # forward to capture activations
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    x = x.cuda(); x.requires_grad_()
    optimizer = Adam([x], lr=lr)
    for _ in range(iteration):
        optimizer.zero_grad(); model(x)
        objective = -layer_activations[:, filterid, :, :].sum()
        objective.backward(); optimizer.step()
    filter_visualizations = x.detach().cpu().squeeze()
    hook_handle.remove()
    return filter_activations, filter_visualizations
```
- hook 捕获指定 CNN 层输出；可视化该滤波器的激活；用梯度上升优化输入以最大化该滤波器响应（滤波器可视化）。

```python
filter_activations, filter_visualizations = filter_explanation(images, model, cnnid=6, filterid=0, iteration=100, lr=0.1)
```
- 展示不同层的滤波器响应与可视化输入。

## Integrated Gradients

```python
class IntegratedGradients():
    def __init__(self, model): self.model = model; self.model.eval()
    def generate_images_on_linear_path(self, input_image, steps):
        return [input_image*step/steps for step in range(steps)]
    def generate_gradients(self, input_image, target_class):
        input_image.requires_grad=True; model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = input_image.grad.data.cpu().numpy()[0]
        return gradients_as_arr
    def generate_integrated_gradients(self, input_image, target_class, steps):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            integrated_grads += self.generate_gradients(xbar_image, target_class)/steps
        return integrated_grads[0]
```
- 实现 IG：在基线到输入的路径上积分梯度。

```python
IG = IntegratedGradients(model)
integrated_grads = [IG.generate_integrated_gradients(img.unsqueeze(0), labels[i], 10) for i,img in enumerate(images.cuda())]
plt.imshow(np.moveaxis(normalize(ig),0,-1))
```
- 展示 IG 热力图。

## BERT 可视化部分（Q21-30）
- 提示使用 https://exbert.net 直接查看注意力。
- 后续代码（未展开）包括安装 transformers、加载 `bert-base-chinese`、PCA/距离工具，用于探索 embedding/层输出。

> 以上逐行解释了 CNN 端的多种可解释性方法实现细节和 BERT 可视化提示。可在实际作业中调整参数（噪声、迭代、滤波器层/ID）以获得更清晰的可视化效果。祝实验顺利！
