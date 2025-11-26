# HW05_ZH 逐行讲解版

面向零基础同学，对 `HW05_ZH.ipynb` 的代码按顺序做中文讲解，帮助理解数据预处理、BPE、fairseq 任务设置、RNN/Transformer 架构、训练与推理流程。代码块后紧跟解释。

## 安装依赖

```python
!pip install 'torch>=1.6.0' editdistance matplotlib sacrebleu sacremoses sentencepiece tqdm wandb
!pip install --upgrade jupyter ipywidgets
```
- 用 pip 安装 PyTorch>=1.6 和评测/分词/可视化工具；升级 jupyter/ipywidgets 以兼容 notebook。

```python
!git clone https://github.com/pytorch/fairseq.git
!cd fairseq && git checkout 9a1c497
!pip install --upgrade ./fairseq/
```
- 克隆指定提交的 fairseq，检出版本 `9a1c497`，本地安装（避免接口变动）。

## 导入模块

```python
import sys, pdb, pprint, logging, os, random
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils import data
import numpy as np
import tqdm.auto as tqdm
from pathlib import Path
from argparse import Namespace
from fairseq import utils
import matplotlib.pyplot as plt
```
- 引入系统、调试、日志、随机、PyTorch、数据工具、NumPy、进度条、路径工具、参数容器、fairseq 工具和绘图。

## 固定随机种子

```python
seed = 73
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```
- 设定随机种子，确保复现；关闭 CuDNN benchmark 以保证确定性。

## 数据集概述
- TED2020 英转繁双语，约 39 万句；测试 4k 句（中文假翻译占位）。

## 下载与解压

```python
data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (...)
file_names = ('ted2020.tgz','test.tgz')
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix/f
    if not path.exists():
        if 'mega' in u:
            !megadl {u} --path {path}
        else:
            !wget {u} -O {path}
    if path.suffix == ".tgz":
        !tar -xvf {path} -C {prefix}
    elif path.suffix == ".zip":
        !unzip -o {path} -d {prefix}
!mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}
!mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}
!mv {prefix/'test.en'} {prefix/'test.raw.en'}
!mv {prefix/'test.zh'} {prefix/'test.raw.zh'}
```
- 设置目录与下载链接；若文件不存在则 wget/megadl 下载；按后缀解压；重命名原始 train/dev/test 文件为统一前缀。

## 语言与路径

```python
src_lang = 'en'
tg_lang = 'zh'
data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'
```
- 指定源/目标语言，设定训练与测试文件前缀。

## 查看前几行

```python
!head {data_prefix+'.'+src_lang} -n 5
!head {data_prefix+'.'+tg_lang} -n 5
```
- 快速检查数据内容。

## 文本清洗函数

```python
def strQ2B(ustring):
    ...  # 全角转半角
def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace('-', '')
        s = re.sub('([.,;!?()"])', r' \1 ', s)
    elif lang == 'zh':
        s = strQ2B(s)
        s = re.sub(r"\([^()]*\)", "", s)
        s = s.replace(' ', '').replace('—','').replace('“','"').replace('”','"').replace('_','')
        s = re.sub('([。,;!?()"~「」])', r' \1 ', s)
    s = ' '.join(s.strip().split())
    return s

def len_s(s, lang):
    return len(s) if lang == 'zh' else len(s.split())
```
- `strQ2B`：全角转半角（空格与字符范围判断）。
- `clean_s`：英文去括号内容、去连字符、标点分隔；中文全角转半角、去括号内容与部分符号、标点分隔；统一多空格。
- `len_s`：中文按字符数，英文按分词数计长度。

```python
def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if cleaned文件已存在则跳过
    打开源/目标文件逐行：
        清洗两种语言句子
        计算长度，按 min/max 限制过滤
        按长度比例 ratio 过滤（避免句长相差过大）
        合格句子写入新的 clean 文件
```
- 双向过滤并输出 clean 版本。

```python
clean_corpus(data_prefix, src_lang, tg_lang)
clean_corpus(test_prefix, src_lang, tg_lang, ratio=-1, min_len=-1, max_len=-1)
```
- 清洗训练/验证及测试（测试不过滤长度）。

```python
!head {data_prefix+'.clean.'+src_lang} -n 5
!head {data_prefix+'.clean.'+tg_lang} -n 5
```
- 检查清洗结果。

## 划分 train/valid

```python
valid_ratio = 0.01
train_ratio = 1 - valid_ratio
...
line_num = 总行数
labels = 打乱的索引
遍历 clean 文件，按比例写入 train.clean/lang 与 valid.clean/lang
```
- 随机 99/1 划分，写入对应文件，已存在则跳过。

## SentencePiece 子词

```python
import sentencepiece as spm
vocab_size = 8000
if 已有模型则跳过
else:
    spm.SentencePieceTrainer.train(
        input=四个 clean 文件,
        model_prefix=..., vocab_size=8000,
        character_coverage=1,
        model_type='unigram',
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )
```
- 训练 unigram 子词模型，词表 8000。

```python
spm_model = spm.SentencePieceProcessor(model_file=...)
in_tag = {'train':'train.clean','valid':'valid.clean','test':'test.raw.clean'}
for split in ['train','valid','test']:
  for lang in [src_lang, tg_lang]:
    若输出已存在则跳过
    否则逐行 encode 为子词并空格连接写出
```
- 对 train/valid/test 进行 BPE 编码生成新文本。

```python
!head {data_dir+'/'+dataset_name+'/train.'+src_lang} -n 5
!head {data_dir+'/'+dataset_name+'/train.'+tg_lang} -n 5
```
- 检查编码后的样本。

## fairseq 预处理成二进制

```python
binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists(): print(存在)
else:
    !python -m fairseq_cli.preprocess \
        --source-lang en --target-lang zh \
        --trainpref train --validpref valid --testpref test \
        --destdir ./DATA/data-bin/ted2020 \
        --joined-dictionary --workers 2
```
- 运行 fairseq 预处理生成二进制数据与词典，复用联合词表。

## 训练配置

```python
config = Namespace(
    datadir="./DATA/data-bin/ted2020",
    savedir="./checkpoints/rnn",
    source_lang="en", target_lang="zh",
    num_workers=2, max_tokens=8192, accum_steps=2,
    lr_factor=2., lr_warmup=4000,
    clip_norm=1.0,
    max_epoch=30, start_epoch=1,
    beam=5, max_len_a=1.2, max_len_b=10,
    post_process="sentencepiece",
    keep_last_epochs=5,
    resume=None,
    use_wandb=False,
)
```
- 定义数据路径、batch token 上限+梯度累积、Noam lr 参数、梯度裁剪、最大 epoch、beam search、生成长度控制、保存/恢复与日志选项。

## 日志与 wandb

```python
logging.basicConfig(...level="INFO"...)
proj = "hw5.seq2seq"
logger = logging.getLogger(proj)
if config.use_wandb: wandb.init(...)
```
- 配置日志格式；可选 wandb 记录。

## CUDA 环境

```python
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
- 检查 CUDA 信息并选择设备。

## 读取数据（fairseq TranslationTask）

```python
from fairseq.tasks.translation import TranslationConfig, TranslationTask

task_cfg = TranslationConfig(
    data=config.datadir,
    source_lang=config.source_lang,
    target_lang=config.target_lang,
    train_subset="train",
    required_seq_len_multiple=8,
    dataset_impl="mmap",
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)
```
- 建立翻译任务，指定数据、语言、子集、对齐长度倍数、mmap 读取。

```python
task.load_dataset(split="train", epoch=1, combine=True)
task.load_dataset(split="valid", epoch=1)
```
- 加载训练/验证集，支持合并回译数据。

```python
sample = task.dataset("valid")[1]
pprint sample
pprint 源/目标字符串(task.dictionary.string(..., post_process="sentencepiece"))
```
- 查看样本与解码后的文本，验证管线。

## 自定义 batch 迭代器

```python
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    return task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(task.max_positions(), max_tokens),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
    )
```
- 创建 batch 迭代器：按 token 数控 batch，过滤过长，种子打乱，支持缓存。

```python
demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
```
- 演示取一个 batch；batch 字典包含 `net_input.src_tokens/prev_output_tokens` 和 `target` 等。

## 模型组件导入

```python
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel
```
- 继承这些基类以兼容 fairseq 的训练/推理（含 beam search）。

## RNN Encoder

```python
class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        ...
        self.embed_tokens = embed_tokens
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, self.num_layers,
                          dropout=args.dropout, batch_first=False, bidirectional=True)
        self.dropout_out_module = nn.Dropout(args.dropout)
        self.padding_idx = dictionary.pad()
```
- 记录嵌入/隐藏维度、层数；双向 GRU；保存 padding 索引用于 mask。

```python
    def combine_bidir(self, outs, bsz):
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()
        return out.view(self.num_layers, bsz, -1)
```
- 将双向输出重排为 [num_layers, batch, hidden*2]。

```python
    def forward(self, src_tokens, **unused):
        bsz, seqlen = src_tokens.size()
        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_module(x)
        x = x.transpose(0, 1)          # [T,B,C]
        h0 = x.new_zeros(2*self.num_layers, bsz, self.hidden_dim)
        x, final_hiddens = self.rnn(x, h0)
        outputs = self.dropout_out_module(x)
        final_hiddens = self.combine_bidir(final_hiddens, bsz)
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return ((outputs, final_hiddens, encoder_padding_mask),)
```
- 前向：嵌入+dropout，GRU 双向输出，合并隐藏状态，生成 padding mask；按 fairseq 约定返回 tuple。

```python
    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple((encoder_out[0].index_select(1, new_order),
                      encoder_out[1].index_select(1, new_order),
                      encoder_out[2].index_select(1, new_order)))
```
- beam search 重排 batch 顺序。

## 注意力层

```python
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, bias=True):
        self.input_proj = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = nn.Linear(input_embed_dim + source_embed_dim, source_embed_dim, bias=bias)
```
- 将 decoder 隐状态投到与 encoder 相同维度，再拼接上下文输出。

```python
    def get_prob(self, x, source_hids, encoder_padding_mask):
        x = self.input_proj(x)                # [B,C]
        attn_scores = torch.einsum('bc,btc->bt', x, source_hids)  # 点积注意力
        attn_scores = attn_scores.masked_fill(encoder_padding_mask.T, -1e4)
        attn_scores = F.softmax(attn_scores, dim=-1)
        return attn_scores
```
- 计算注意力权重，mask 掉 padding。

```python
    def forward(self, x, source_hids, encoder_padding_mask):
        attn_scores = self.get_prob(x, source_hids, encoder_padding_mask)
        context = torch.einsum('bt,btc->bc', attn_scores, source_hids)
        x = torch.cat([x, context], dim=-1)
        x = self.output_proj(x)
        return x, attn_scores
```
- 按权重加权 encoder 输出得到上下文，拼接后线性投影返回上下文与注意力。

## RNN Decoder

```python
class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(self.embed_dim + (0 if no_encoder_attn else args.encoder_ffn_embed_dim*2),
                          self.hidden_dim, self.num_layers,
                          dropout=args.dropout, batch_first=False)
        self.dropout_out_module = nn.Dropout(args.dropout)
        self.attention = None if no_encoder_attn else AttentionLayer(self.hidden_dim, args.encoder_ffn_embed_dim*2)
        self.padding_idx = dictionary.pad()
```
- GRU 输入含嵌入和可选上下文；可关闭注意力。

```python
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        source, encoder_hiddens, padding_mask = encoder_out
        encoder_padding_mask = padding_mask.t()
        bsz, seqlen = prev_output_tokens.size()
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)
        x = x.transpose(0,1)
        attn_scores = None
        if incremental_state is None:
            hidden_state = encoder_hiddens  # 用 encoder 最终隐状态做初始
            cached_source = source
            cached_mask = encoder_padding_mask
            start = 0
        else:
            cached_source, cached_mask, hidden_state = self._get_cache('cache', incremental_state)
            start = cached_source.size(1)
            cached_source = torch.cat([cached_source, source], dim=1)
            cached_mask = torch.cat([cached_mask, encoder_padding_mask], dim=1)
        if self.attention:
            cache = []
            for i in range(x.size(0)):
                if i == 0 and incremental_state is not None:
                    hidden = hidden_state
                else:
                    _, hidden = self.rnn(x[i:i+1], hidden_state)
                    hidden_state = hidden
                query = hidden_state[-1]              # 取最顶层隐藏
                query, attn_score = self.attention(query, cached_source, cached_mask)
                cache.append(attn_score)
                _, hidden = self.rnn(torch.cat([x[i:i+1], query.unsqueeze(0)], dim=-1), hidden_state)
                hidden_state = hidden
            x = torch.cat([h for h in cache])         # 用 attention 的输出替换 rnn 输出来对齐
            attn_scores = torch.stack(cache)
        else:
            x, hidden_state = self.rnn(x, hidden_state)
        x = self.dropout_out_module(x)
        x = x.transpose(0,1)
        self._set_cache('cache', (cached_source, cached_mask, hidden_state), incremental_state)
        return x, {"attn": attn_scores}
```
- 支持 incremental_state（用于解码时缓存）；使用 encoder 隐状态初始化；按步注意力、GRU 更新；缓存注意力；返回输出与注意力。

```python
    def reorder_incremental_state(self, incremental_state, new_order):
        cached_source, cached_mask, hidden_state = self._get_cache('cache', incremental_state)
        cached_source = cached_source.index_select(0,new_order)
        cached_mask = cached_mask.index_select(0,new_order)
        hidden_state = hidden_state.index_select(1,new_order)
        self._set_cache('cache',(cached_source,cached_mask,hidden_state), incremental_state)
```
- beam 搜索时重排缓存。

## Encoder-Decoder 模型封装

```python
class Seq2Seq(FairseqEncoderDecoderModel):
    @classmethod
    def build_model(cls, args, task):
        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
        if args.share_all_embeddings:
            # share encoder/decoder embeddings
            embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
            encoder_embed_tokens = decoder_embed_tokens = embed_tokens
        else:
            encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
            decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
        encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
        decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)
```
- 根据任务词典与 args 构造嵌入（可共享），创建 encoder/decoder 并返回模型。

```python
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens=src_tokens)
        logits, extra = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return logits, extra
```
- 前向：编码 + 解码，输出 logits 与附加信息。

## 模型参数

```python
model = Seq2Seq.build_model(args, task)
```
- 通过 fairseq Registry 构建（实际 notebook 会在训练脚本中调用）。

## Noam 学习率调度

```python
def get_rate(i, d_model, factor, warmup):
    return factor * (d_model ** (-0.5) * min(i**(-0.5), i*warmup**(-1.5)))
```
- 经典 transformer Noam lr 公式。

```python
def get_scheduler(optimizer, d_model, factor, warmup, steps):
    lrs = [get_rate(i+1,d_model,factor,warmup) for i in range(steps)]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lrs[min(step,len(lrs)-1)]/lrs[0])
```
- 预生成学习率列表，用 LambdaLR 应用。

## 训练循环（伪代码解析）

核心步骤（源码中按 fairseq 接口实现）：
- 初始化模型、Criterion（label smoothed cross entropy）、优化器（Adam）、调度器（Noam）。
- `for epoch in range(start_epoch, max_epoch+1):`
  - `task.load_dataset(split="train", epoch=epoch)`
  - 创建 batch 迭代器 `iterator = load_data_iterator(...)`
  - `for step, samples in enumerate(iterator.next_epoch_itr(...))`:
    - `model.train(); optimizer.zero_grad()`
    - `logits, extra = model(**samples['net_input'])`
    - 计算损失 `criterion(logits, samples['target']) / accum_steps`
    - `loss.backward(); grad_norm = clip_grad_norm_(...)`
    - 若达到累积步数则 `optimizer.step(); scheduler.step(); optimizer.zero_grad()`
    - 日志记录 step/epoch/loss/ppl/lr/grad_norm
    - 定期保存 checkpoint（仅保留最近 keep_last_epochs 个）
  - 验证：`model.eval(); for valid_batch in valid_iterator: 计算 loss/bleu`
  - 如果 `use_wandb` 则记录指标。

> 训练具体实现位于 notebook 末尾的训练 cell，遵循上述逻辑，利用 fairseq 的 batch 字典（含 `src_tokens`, `src_lengths`, `prev_output_tokens`, `target`）。

## 推理（生成翻译）

基本流程（对应 notebook 生成段）：
1. 加载最佳 checkpoint，`model.eval()`。
2. 构造测试集迭代器 `task.get_batch_iterator(split="test", ...)`。
3. 对每个 batch：
   - 调用 `task.inference_step(generator, models, sample)`，其中 `generator` 是 beam search（`SequenceGenerator`），`models` 包含当前模型。
   - 得到生成的 token 序列，去掉 `<s>/<\s>`，用 `task.target_dictionary.string(..., post_process="sentencepiece")` 解码为文本。
4. 将生成的中文句子写入输出文件 `output.zh` 或 CSV。

## Back-translation（提示）
- 作业 TODO：用模型把中文翻译为英文生成伪双语，再合并回训练集，提升性能；流程与上面类似，只是交换源/目标。

> 以上讲解覆盖 notebook 中主要代码块，逐行解释了数据清洗、BPE、fairseq 数据管线、RNN 编解码器、注意力、训练与推理的实现要点，便于理解 Seq2Seq 与 Transformer 任务的完整流程。若要进一步提升，可尝试改用 TransformerEncoder/Decoder、增加层数或实现回译。祝学习顺利！
