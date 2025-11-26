# hw11_domain_adaptation_(en) 逐行讲解版

面向零基础同学，对 `hw11_domain_adaptation_(en).ipynb` 的代码按顺序解释，帮助理解 Domain Adversarial Training (DaNN) 实现。代码块后紧跟说明。

## 场景与目标
- 源域：带标签的真实照片；目标域：无标签的手绘涂鸦（分布不同）。
- 训练：使用源域标签做分类，同时通过域对抗让特征分布在源/目标上对齐，使分类器可泛化到目标域。

## 数据下载与可视化

```python
!gdown --id '1P4fGNb9JhJj8W0DA_Qrp7mbrRHfF5U_f' --output real_or_drawing.zip
!unzip real_or_drawing.zip
```
- 下载并解压包含 `train_data`（源域）和 `test_data`（目标域）的目录，类别均衡。

## 数据加载与增强

```python
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms

source_transform = transforms.Compose([
    transforms.Grayscale(),            # 转灰度
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(), # 0.5 概率水平翻转
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader   = DataLoader(target_dataset, batch_size=128, shuffle=False)
```
- 源域用较强增强（翻转/旋转）以防过拟合；目标域仅缩放、灰度、张量化。

## 模型

```python
class FeatureExtractor(nn.Module):
    # VGG 风格 5 段卷积+BN+ReLU+MaxPool，将 1x32x32 -> 512x1x1，最后 squeeze 得到 512 维特征。

class LabelPredictor(nn.Module):
    # 全连接 512->512->512->10，ReLU，中间无归一化，输出10类 logits。

class DomainClassifier(nn.Module):
    # 全连接堆叠 5 层 512 宽度 + BN + ReLU，最后输出 1 维域判别 logits（源=1, 目标=0）。
```
- 特征提取共享，标签预测与域分类各自线性头。

## 损失与优化器

```python
feature_extractor = FeatureExtractor().cuda()
label_predictor   = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion  = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())
```
- 源域分类用交叉熵；域判别用二分类 BCE（带 logits）。

## 训练循环（DaNN）

```python
def train_epoch(source_dataloader, target_dataloader, lamb):
    running_D_loss = running_F_loss = 0
    total_hit = total_num = 0
    for (source_data, source_label), (target_data, _) in zip(source_dataloader, target_dataloader):
        source_data, source_label, target_data = source_data.cuda(), source_label.cuda(), target_data.cuda()
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros(len(mixed_data), 1).cuda()
        domain_label[:len(source_data)] = 1  # 源域标签=1，目标=0

        # Step 1: 训练域分类器（冻结特征）
        feature = feature_extractor(mixed_data)
        domain_logits = domain_classifier(feature.detach())     # detach 防止回传到 F
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss += loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2: 训练特征提取器+标签预测器（对抗域分类）
        class_logits = label_predictor(feature[:len(source_data)])  # 仅源域有标签
        domain_logits = domain_classifier(feature)                  # 不 detach，让梯度回到 F
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss += loss.item()
        loss.backward()
        optimizer_F.step(); optimizer_C.step()

        optimizer_D.zero_grad(); optimizer_F.zero_grad(); optimizer_C.zero_grad()
        total_hit += (class_logits.argmax(dim=1) == source_label).sum().item()
        total_num += len(source_data)
    return running_D_loss/(i+1), running_F_loss/(i+1), total_hit/total_num
```
- 域分类器先学分辨源/目标；特征提取器用 “分类损失 - λ*域损失” 使特征既能分类又让域判别困难（对抗训练类似 GAN）。
- 注意混合源/目标再做 BN，避免统计量错位。

### 训练主循环

```python
for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(..., lamb=0.1)
    torch.save(feature_extractor.state_dict(), 'extractor_model.bin')
    torch.save(label_predictor.state_dict(), 'predictor_model.bin')
    print(...)
```
- 训练 200 轮，保存权重，打印域损失/总损失/源域准确率。可按原论文使用自适应 λ，或继续训练以稳定。

## 推理与提交

```python
result = []
label_predictor.eval(); feature_extractor.eval()
for test_data, _ in test_dataloader:
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    result.append(class_logits.argmax(dim=1).cpu().numpy())
result = np.concatenate(result)
pd.DataFrame({'id': np.arange(len(result)), 'label': result}).to_csv('DaNN_submission.csv', index=False)
```
- 用目标域数据前向预测类别，生成提交 CSV。

## 关键点与可调整项
- `lamb` 控制分类与域对齐的权衡，可随 epoch 自适应。
- 源/目标数据均衡，可利用平衡采样或同时取 batch。
- 增强：可调整源域增强强度；目标域可加弱增强。
- 优化：学习率、权重衰减、梯度裁剪、调大 epoch。
- 域分类器/特征提取器容量可调整；可加入 Gradient Reversal Layer 替代手动两步训练。

> 以上覆盖 notebook 主要代码行，解释了数据处理、模型结构、DaNN 训练策略及生成提交的流程。祝实验顺利！***
