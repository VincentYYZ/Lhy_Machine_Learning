# ML2021_HW4 逐行讲解版

面向零基础同学，对 `ML2021_HW4.ipynb` 中的代码逐段解释。任务：用 Transformer 做说话人分类。代码块后紧跟解释，顺序与 notebook 保持一致。

## 下载数据

```python
# Google Drive 备用
# !gdown --id '1T0RPnu-Sg5eIPwQPfYysipfcz81MnsYe' --output Dataset.zip
# !unzip Dataset.zip

# Dropbox 分片下载并解压
!wget https://www.dropbox.com/s/vw324newiku0sz0/Dataset.tar.gz.aa?dl=0
!wget https://www.dropbox.com/s/z840g69e7lnkayo/Dataset.tar.gz.ab?dl=0
!wget https://www.dropbox.com/s/hl081e1ggonio81/Dataset.tar.gz.ac?dl=0
!wget https://www.dropbox.com/s/fh3zd8ow668c4th/Dataset.tar.gz.ad?dl=0
!wget https://www.dropbox.com/s/ydzygoy2pv6gw9d/Dataset.tar.gz.ae?dl=0
!cat Dataset.tar.gz.* | tar zxvf -
```
- 五个分片下载到当前目录，`cat ... | tar zxvf -` 拼接并解压得到 `Dataset/`（包含 mel 特征 `.pt`、`metadata.json`、`testdata.json`、`mapping.json`）。

## 数据集类与切分

```python
import os, json, torch, random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class myDataset(Dataset):
  def __init__(self, data_dir, segment_len=128):
    self.data_dir = data_dir
    self.segment_len = segment_len
    mapping = json.load((Path(data_dir) / "mapping.json").open())
    self.speaker2id = mapping["speaker2id"]
    metadata = json.load(open(Path(data_dir) / "metadata.json"))["speakers"]
    self.speaker_num = len(metadata.keys())
    self.data = []
    for speaker in metadata.keys():
      for utterances in metadata[speaker]:
        self.data.append([utterances["feature_path"], self.speaker2id[speaker]])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    feat_path, speaker = self.data[index]
    mel = torch.load(os.path.join(self.data_dir, feat_path))
    if len(mel) > self.segment_len:
      start = random.randint(0, len(mel) - self.segment_len)
      mel = torch.FloatTensor(mel[start:start+self.segment_len])
    else:
      mel = torch.FloatTensor(mel)
    speaker = torch.FloatTensor([speaker]).long()
    return mel, speaker

  def get_speaker_number(self):
    return self.speaker_num
```
- 读取 `mapping.json` 得到说话人 id 映射；`metadata.json` 列出各说话人语音特征路径。
- `self.data` 收集 `(特征路径, 说话人id)`。
- `__getitem__`：加载 mel 频谱（已预处理），长度>segment_len 时随机裁一段；否则原长；返回 mel 张量与 label（long）。
- `get_speaker_number`：返回类别数。

```python
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
  mel, speaker = zip(*batch)
  mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # 对齐到同长
  return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, batch_size, n_workers):
  dataset = myDataset(data_dir)
  speaker_num = dataset.get_speaker_number()
  trainlen = int(0.9 * len(dataset))
  trainset, validset = random_split(dataset, [trainlen, len(dataset) - trainlen])
  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True,
                            num_workers=n_workers, pin_memory=True, collate_fn=collate_batch)
  valid_loader = DataLoader(validset, batch_size=batch_size, drop_last=True,
                            num_workers=n_workers, pin_memory=True, collate_fn=collate_batch)
  return train_loader, valid_loader, speaker_num
```
- `collate_batch`：批内 padding，填充值为极小 -20（log 概率）；返回对齐的 mel 与标签。
- `get_dataloader`：9:1 划分训练/验证，构造对应 DataLoader。

## 模型：Transformer 编码器

```python
import torch.nn as nn

class Classifier(nn.Module):
  def __init__(self, d_model=80, n_spks=600, dropout=0.1):
    super().__init__()
    self.prenet = nn.Linear(40, d_model)  # 40 维 mel 投影到 d_model
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    # TODO: 可替换为 Conformer 进一步提升
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.ReLU(),
      nn.Linear(d_model, n_spks),
    )

  def forward(self, mels):
    # mels: [B, T, 40]
    out = self.prenet(mels)           # [B, T, d_model]
    out = out.permute(1, 0, 2)        # [T, B, d_model] 供 Transformer 使用
    out = self.encoder_layer(out)     # 单层 encoder（可堆叠）
    out = out.transpose(0, 1)         # 回到 [B, T, d_model]
    stats = out.mean(dim=1)           # 时序平均池化
    out = self.pred_layer(stats)      # [B, n_spks] logits
    return out
```
- 前端线性将 40 维特征投影；一层 `TransformerEncoderLayer` 处理序列；时间维平均池化得到句级表示；全连接映射到说话人类别。

## 余弦衰减+warmup 学习率

```python
import math
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, last_epoch=-1):
  def lr_lambda(current_step):
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
  return LambdaLR(optimizer, lr_lambda, last_epoch)
```
- 前 `num_warmup_steps` 学习率线性升到基准；之后按半个余弦波衰减到 0。

## 前向封装、验证函数

```python
def model_fn(batch, model, criterion, device):
  mels, labels = batch
  mels, labels = mels.to(device), labels.to(device)
  outs = model(mels)
  loss = criterion(outs, labels)
  preds = outs.argmax(1)
  accuracy = torch.mean((preds == labels).float())
  return loss, accuracy
```
- 封装一次前向，返回 loss 与准确率。

```python
from tqdm import tqdm
def valid(dataloader, model, criterion, device):
  model.eval()
  running_loss = running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item(); running_accuracy += accuracy.item()
    pbar.update(dataloader.batch_size)
    pbar.set_postfix(loss=f"{running_loss/(i+1):.2f}", accuracy=f"{running_accuracy/(i+1):.2f}")
  pbar.close()
  model.train()
  return running_accuracy / len(dataloader)
```
- 验证遍历 DataLoader，累积平均损失/准确率，进度条显示；结束后返回平均准确率，模型切回 train。

## 训练主程序

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

def parse_args():
  return {
    "data_dir": "./Dataset",
    "save_path": "model.ckpt",
    "batch_size": 32,
    "n_workers": 8,
    "valid_steps": 2000,
    "warmup_steps": 1000,
    "save_steps": 10000,
    "total_steps": 70000,
  }
```
- 配置项：数据路径、batch 大小、线程数、验证/保存/总步数、warmup 步数等。

```python
def main(data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps, total_steps, save_steps):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!", flush=True)

  model = Classifier(n_spks=speaker_num).to(device)
  criterion = nn.CrossEntropyLoss()
  optimizer = AdamW(model.parameters(), lr=1e-3)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!", flush=True)

  best_accuracy = -1.0
  best_state_dict = None
  pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

  for step in range(total_steps):
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss, accuracy = model_fn(batch, model, criterion, device)
    batch_loss, batch_accuracy = loss.item(), accuracy.item()
    loss.backward()
    optimizer.step(); scheduler.step(); optimizer.zero_grad()

    pbar.update()
    pbar.set_postfix(loss=f"{batch_loss:.2f}", accuracy=f"{batch_accuracy:.2f}", step=step + 1)

    if (step + 1) % valid_steps == 0:
      pbar.close()
      valid_accuracy = valid(valid_loader, model, criterion, device)
      if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        best_state_dict = model.state_dict()
      pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    if (step + 1) % save_steps == 0 and best_state_dict is not None:
      torch.save(best_state_dict, save_path)
      pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

  pbar.close()

if __name__ == "__main__":
  main(**parse_args())
```
- 取 GPU/CPU 设备，加载数据；创建模型、损失、AdamW 优化器、余弦 warmup 调度。
- 主循环：不断取 batch（耗尽则重置迭代器），前向/反传/步进调度；进度条显示当前 loss/acc。
- 每 `valid_steps` 做一次验证，保存最佳权重到内存。
- 每 `save_steps` 把当前最佳权重存盘 `model.ckpt`。

## 推理数据集与主程序

```python
class InferenceDataset(Dataset):
  def __init__(self, data_dir):
    testdata_path = Path(data_dir) / "testdata.json"
    metadata = json.load(testdata_path.open())
    self.data_dir = data_dir
    self.data = metadata["utterances"]
  def __len__(self): return len(self.data)
  def __getitem__(self, index):
    utterance = self.data[index]
    feat_path = utterance["feature_path"]
    mel = torch.load(os.path.join(self.data_dir, feat_path))
    return feat_path, mel

def inference_collate_batch(batch):
  feat_paths, mels = zip(*batch)
  return feat_paths, torch.stack(mels)
```
- 读取 `testdata.json` 里的测试语音特征路径；逐条加载 mel；collate 直接堆叠张量。

```python
def parse_args():
  return {
    "data_dir": "./Dataset",
    "model_path": "./model.ckpt",
    "output_path": "./output.csv",
  }

def main(data_dir, model_path, output_path):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")
  mapping = json.load((Path(data_dir) / "mapping.json").open())

  dataset = InferenceDataset(data_dir)
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                          num_workers=8, collate_fn=inference_collate_batch)
  print(f"[Info]: Finish loading data!", flush=True)

  speaker_num = len(mapping["id2speaker"])
  model = Classifier(n_spks=speaker_num).to(device)
  model.load_state_dict(torch.load(model_path))
  model.eval()
  print(f"[Info]: Finish creating model!", flush=True)

  results = [["Id", "Category"]]
  for feat_paths, mels in tqdm(dataloader):
    with torch.no_grad():
      outs = model(mels.to(device))
      preds = outs.argmax(1).cpu().numpy()
      for feat_path, pred in zip(feat_paths, preds):
        results.append([feat_path, mapping["id2speaker"][str(pred)]])
  
  with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(results)

if __name__ == "__main__":
  main(**parse_args())
```
- 加载映射、构造推理 DataLoader（不打乱、批量 1）。
- 重新实例化与训练同结构的模型，加载保存的权重，设为 eval。
- 对每条测试 mel 前向得到 logits，取 argmax 类别，再映射回说话人字符串 `id2speaker`。
- 收集为 CSV，首行 `Id,Category`，后续每行 `特征路径,预测说话人`。

> 以上讲解涵盖 notebook 全部代码，帮助理解数据载入、Transformer 模型、学习率调度、训练/验证与推理输出。进阶可在 TODO 处改用 Conformer、增加 encoder 层数或改进超参以提升表现。祝学习顺利！
