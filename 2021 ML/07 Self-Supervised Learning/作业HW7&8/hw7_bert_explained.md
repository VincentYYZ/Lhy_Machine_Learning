# hw7_bert 逐行讲解版

面向零基础同学，对 `hw7_bert.ipynb` 代码按顺序逐段解释，帮助理解中文抽取式问答的 BERT 微调流程。代码块后紧跟说明。

## 任务描述
- 输入：问题+段落，输出：在段落中的答案片段（起止位置）。
- 目标：使用 HuggingFace transformers 微调中文 BERT 做抽取式 QA；练习超参调整、学习率调度、梯度累积/混合精度、后处理等。

## 下载数据

```python
!gdown --id '1znKmX08v9Fygp-dgwo7BKiLIf2qL1FH1' --output hw7_data.zip
# 备用链接在注释
!unzip -o hw7_data.zip
!nvidia-smi
```
- 下载并解压官方数据；查看 GPU 型号。

## 安装 transformers

```python
!pip install transformers==4.5.0
```
- 安装指定版本的 HuggingFace transformers（可按需改版本）。

## 导入与设备、种子

```python
import json, numpy as np, random, torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
same_seeds(0)
```
- 引入训练所需包；选择 GPU/CPU；固定随机种子。

### 可选混合精度

```python
fp16_training = False
if fp16_training:
    !pip install accelerate==0.2.0
    from accelerate import Accelerator
    accelerator = Accelerator(fp16=True)
    device = accelerator.device
```
- 将 `fp16_training=True` 时启用 `accelerate` 进行混合精度/分布式封装，更新 device。

## 加载预训练模型与分词器

```python
model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")
```
- 中文 BERT + QA 头；Fast tokenizer 便于 offset 对齐。

## 读入数据

```python
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

train_questions, train_paragraphs = read_data("hw7_train.json")
dev_questions, dev_paragraphs = read_data("hw7_dev.json")
test_questions, test_paragraphs = read_data("hw7_test.json")
```
- JSON 包含 `questions` 列表（含 id、paragraph_id、question_text、answer_text、answer_start/end）和 `paragraphs` 列表（索引与 paragraph_id 对应）。

## 预分词（问题与段落分开）

```python
train_questions_tokenized = tokenizer([...question_text...], add_special_tokens=False)
dev_questions_tokenized = ...
test_questions_tokenized = ...
train_paragraphs_tokenized = tokenizer(train_paragraphs, add_special_tokens=False)
dev_paragraphs_tokenized = ...
test_paragraphs_tokenized = ...
```
- 分别对问题/段落做分词，不加特殊符号，后续在 Dataset 组合。

## 数据集与 DataLoader

```python
class QA_Dataset(Dataset):
    def __init__(...):
        self.max_question_len = 40
        self.max_paragraph_len = 150
        self.doc_stride = 150   # TODO 可调整
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1  # [CLS] Q [SEP] P [SEP]
```
- 设定最大长度与文档滑窗步长（`doc_stride` 可调）。

### __getitem__ 训练/验证逻辑

训练：
- 根据 QA 对应的段落分词结果，利用 `char_to_token` 将答案的字符起止位置映射到段落 token 起止。
- 以答案中点为中心截取一段长度 `max_paragraph_len` 的窗口，防止答案被裁掉。
- 组装输入：`[CLS] + Q + [SEP] + 段落窗口 + [SEP]`，并重新计算答案起止在该窗口的索引（加上 question 部分长度，减去窗口起点）。
- 调用 `padding` 生成 `input_ids/token_type_ids/attention_mask`，返回张量和答案起止位置。

验证/测试：
- 将段落按 `doc_stride` 滑窗切成多个窗口；对每个窗口与问题组装输入，padding 后返回三类张量列表（形状 [num_windows, max_seq_len]）。

### padding 函数

```python
padding_len = max_seq_len - len(question_part) - len(paragraph_part)
input_ids = question_part + paragraph_part + [0]*padding_len
token_type_ids = [0]*len(question_part) + [1]*len(paragraph_part) + [0]*padding_len
attention_mask = [1]*(len(question_part)+len(paragraph_part)) + [0]*padding_len
```
- 用 0 pad；`token_type_ids` 区分 Q/P；mask 0 表示 padding。

### DataLoader

```python
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True)
dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)
```
- dev/test 固定 batch=1（内部含多窗口）。

## 后处理 evaluate（需修正的 TODO）

```python
def evaluate(data, output):
    answer = ''
    max_prob = -inf
    num_of_windows = data[0].shape[1]
    for k in range(num_of_windows):
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        prob = start_prob + end_prob
        if prob > max_prob:
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index : end_index + 1])
    return answer.replace(' ','')
```
- 当前做法：遍历窗口取 start/end 各自最大，按概率和选最佳窗口，再解码、去空格。
- Bug/改进点：未确保 `start_index <= end_index` 且窗口内 span 合理；未做指针组合搜索/约束；doc_stride 叠窗可能截断；可加入 `token_type_ids` mask、长度筛选或 beam 组合。

## 训练循环

```python
num_epoch = 1
validation = True
logging_step = 100
learning_rate = 1e-4
optimizer = AdamW(model.parameters(), lr=learning_rate)
if fp16_training:
    model, optimizer, train_loader = accelerator.prepare(...)

model.train()
for epoch in range(num_epoch):
    step = 1; train_loss = train_acc = 0
    for data in tqdm(train_loader):
        data = [i.to(device) for i in data]
        output = model(input_ids=data[0], token_type_ids=data[1],
                       attention_mask=data[2], start_positions=data[3], end_positions=data[4])
        start_index = torch.argmax(output.start_logits, dim=1)
        end_index = torch.argmax(output.end_logits, dim=1)
        train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
        train_loss += output.loss
        if fp16_training: accelerator.backward(output.loss)
        else: output.loss.backward()
        optimizer.step(); optimizer.zero_grad(); step += 1
        # TODO: 线性学习率衰减
        if step % logging_step == 0:
            print(...)
            train_loss = train_acc = 0
```
- 前向返回 loss（因提供了 start/end）；算准确率需起止都正确；反传更新；留有 TODO 实现线性 lr 衰减（可用 scheduler 或手工调整）。

### 验证

```python
if validation:
    model.eval()
    with torch.no_grad():
        dev_acc = 0
        for i, data in enumerate(tqdm(dev_loader)):
            output = model(input_ids=data[0].squeeze().to(device), token_type_ids=... , attention_mask=...)
            dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
    print(f"acc = {dev_acc / len(dev_loader):.3f}")
    model.train()
```
- 逐条验证，预测文本与真值完全匹配计对；注意 `squeeze` 还原窗口维度。

### 保存

```python
model_save_dir = "saved_model"
model.save_pretrained(model_save_dir)
```
- 保存权重和配置，可用 `from_pretrained` 加载。

## 测试与提交

```python
result = []
model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        output = model(input_ids=data[0].squeeze(0).to(device),
                       token_type_ids=data[1].squeeze(0).to(device),
                       attention_mask=data[2].squeeze(0).to(device))
        result.append(evaluate(data, output))

with open("result.csv",'w') as f:
    f.write("ID,Answer\n")
    for i, test_question in enumerate(test_questions):
        f.write(f"{test_question['id']},{result[i].replace(',','')}\n")
```
- 对测试集推理，后处理得到答案，写 CSV（去掉逗号）。

> 以上覆盖 notebook 所有代码行，逐段解释了数据处理、BERT QA 模型、训练/验证/测试与需要完成的 TODO（doc_stride、预处理屏蔽泄漏、后处理 span 约束、线性 LR 衰减）。可按提示改进以提升准确率。祝实验顺利！
