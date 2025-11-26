# HW05 逐行讲解版（英文原版代码说明）

对 `HW05.ipynb` 的代码按顺序解释，帮助理解英文版 Seq2Seq 翻译任务（与 ZH 版结构相同）。涵盖依赖安装、数据预处理、BPE、fairseq 任务、模型、训练与推理。

## Install dependencies

```python
!pip install 'torch>=1.6.0' editdistance matplotlib sacrebleu sacremoses sentencepiece tqdm wandb
!pip install --upgrade jupyter ipywidgets
```
- Install PyTorch and eval/tokenization utilities; upgrade Jupyter widgets.

```python
!git clone https://github.com/pytorch/fairseq.git
!cd fairseq && git checkout 9a1c497
!pip install --upgrade ./fairseq/
```
- Clone fairseq, checkout fixed commit, install locally to avoid API drift.

## Imports

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
- Standard/system tools, PyTorch core, dataloaders, NumPy, progress bar, paths, config container, fairseq helpers, plotting.

## Set random seed

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
- Fix randomness for reproducibility; turn off CuDNN autotune to stay deterministic.

## Data overview
- TED2020 English→Traditional Chinese parallel corpus (~394k after cleaning); test has 4k sentences (Chinese hidden).

## Download and extract

```python
data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (... two download links ...)
file_names = ('ted2020.tgz','test.tgz')
prefix = Path(data_dir).absolute() / dataset_name

prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix/f
    if not path.exists():
        if 'mega' in u: !megadl {u} --path {path}
        else: !wget {u} -O {path}
    if path.suffix == ".tgz": !tar -xvf {path} -C {prefix}
    elif path.suffix == ".zip": !unzip -o {path} -d {prefix}
!mv raw.en -> train_dev.raw.en; raw.zh -> train_dev.raw.zh; test.en -> test.raw.en; test.zh -> test.raw.zh
```
- Download two archives (train/dev and test), extract, and rename to unified prefixes.

## Language setup

```python
src_lang = 'en'; tgt_lang = 'zh'
data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'
```
- Source/target codes and path prefixes.

```python
!head ...train_dev.raw.en -n 5
!head ...train_dev.raw.zh -n 5
```
- Peek at raw lines.

## Cleaning helpers

```python
def strQ2B(ustring): ... # full-width to half-width

def clean_s(s, lang):
    if lang == 'en':
        remove brackets, hyphen; space out punctuation
    elif lang == 'zh':
        Q2B, remove brackets/spaces/dashes/quotes/underscores, normalize punctuation, space out punctuation
    strip & collapse spaces
    return s

def len_s(s, lang):
    return len(s) if lang=='zh' else len(s.split())
```
- Normalize text per language and get length (chars vs tokens).

```python
def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):
    if cleaned files exist: skip
    open src/tgt, read line pairs:
        clean both
        length filters (min/max) and length ratio filter
        write accepted pairs to .clean.{l1}/{l2}
```
- Bilingual cleaning with length/ration filters.

```python
clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)
```
- Clean train/dev and test (test without filters).

```python
!head train_dev.raw.clean.en -n 5
!head train_dev.raw.clean.zh -n 5
```
- Inspect cleaned results.

## Train/valid split

```python
valid_ratio = 0.01
train_ratio = 1 - valid_ratio
if split files exist: skip
else:
    line_num = count lines
    labels = shuffled indices
    for each lang:
        open train.clean.lang & valid.clean.lang
        iterate clean file, assign to train/valid by shuffled order and ratio
```
- Random 99/1 split into train/valid.

## SentencePiece BPE/unigram

```python
import sentencepiece as spm
vocab_size = 8000
if model exists: skip
else:
    spm.SentencePieceTrainer.train(
        input=all train/valid clean files (en/zh),
        model_prefix=spm{vocab_size},
        vocab_size=8000,
        character_coverage=1,
        model_type='unigram',  # bpe also possible
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )
```
- Train unigram tokenizer with shared vocab.

```python
spm_model = spm.SentencePieceProcessor(model_file=...)
in_tag = {'train':'train.clean','valid':'valid.clean','test':'test.raw.clean'}
for split in ['train','valid','test']:
  for lang in [en, zh]:
    out_path exists -> skip
    else encode each line to subwords, join with spaces, write to split.lang
```
- Encode all splits into subword tokenized text.

```python
!head .../train.en -n 5
!head .../train.zh -n 5
```
- Check tokenized examples.

## fairseq binarization

```python
binpath = ./DATA/data-bin/ted2020
if exists: print skip
else:
    python -m fairseq_cli.preprocess \
      --source-lang en --target-lang zh \
      --trainpref train --validpref valid --testpref test \
      --destdir DATA/data-bin/ted2020 \
      --joined-dictionary --workers 2
```
- Convert text to indexed binary format and build shared dictionary.

## Experiment config

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
- Data paths, batching/token limits, Noam LR settings, grad clip, epochs, beam search length control, checkpointing.

## Logging and wandb

```python
logging.basicConfig(...level="INFO"...)
logger = logging.getLogger("hw5.seq2seq")
if config.use_wandb: wandb.init(...)
```
- Set logging; optional wandb tracking.

## CUDA environment

```python
cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```
- Inspect CUDA; pick device.

## Load datasets via TranslationTask

```python
from fairseq.tasks.translation import TranslationConfig, TranslationTask
task_cfg = TranslationConfig(
    data=config.datadir, source_lang=config.source_lang, target_lang=config.target_lang,
    train_subset="train", required_seq_len_multiple=8, dataset_impl="mmap", upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)
task.load_dataset(split="train", epoch=1, combine=True)
task.load_dataset(split="valid", epoch=1)
sample = task.dataset("valid")[1]
print decoded source/target via task.{source,target}_dictionary.string(..., post_process="sentencepiece")
```
- Configure translation task, load train/valid, inspect a sample.

## Batch iterator helper

```python
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
    return task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        max_positions=utils.resolve_max_positions(task.max_positions(), max_tokens),
        ignore_invalid_inputs=True,
        seed=seed, num_workers=num_workers, epoch=epoch,
        disable_iterator_cache=not cached,
    )
demo_epoch_obj = load_data_iterator(... max_tokens=20, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
```
- Create token-bounded iterator; demo retrieving a batch (dict with `net_input` and `target`).

## Model components

```python
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel
```
- Base classes for fairseq-compatible encoder/decoder/model.

### RNNEncoder
- Embedding + bidirectional GRU with dropout; combine bidir states; output tuple `(outputs, final_hiddens, encoder_padding_mask)`; includes `reorder_encoder_out` for beam search.

### AttentionLayer
- Projects decoder hidden to encoder dim, dot-product attention with mask, softmax to get weights, weighted sum context, concat with query, project to output; returns context-enhanced vector and attn scores.

### RNNDecoder
- Embedding + GRU; optional attention:
  - If attention enabled, step through tokens, use encoder final states as init hidden, attend over encoder outputs (cached for incremental decoding), feed context+embedding into GRU, collect attn scores.
  - Supports incremental_state cache for autoregressive decoding and `reorder_incremental_state` for beam search.
- Returns decoder outputs and `{"attn": attn_scores}`.

### Seq2Seq wrapper

```python
class Seq2Seq(FairseqEncoderDecoderModel):
    @classmethod
    def build_model(cls, args, task):
        build shared or separate embeddings; create RNNEncoder/Decoder; return model
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        encoder_out = self.encoder(src_tokens=src_tokens)
        logits, extra = self.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)
        return logits, extra
```
- Fairseq-compatible encoder-decoder; can switch to Transformer by replacing modules.

## Noam LR schedule

```python
def get_rate(i, d_model, factor, warmup):
    return factor * (d_model ** -0.5 * min(i**-0.5, i*warmup**-1.5))

def get_scheduler(optimizer, d_model, factor, warmup, steps):
    lrs = [get_rate(i+1, d_model, factor, warmup) for i in range(steps)]
    return LambdaLR(optimizer, lr_lambda=lambda step: lrs[min(step,len(lrs)-1)]/lrs[0])
```
- Implements Transformer Noam schedule with warmup then inverse sqrt decay.

## Training loop (as implemented in notebook)
- Initialize model, criterion (label-smoothed cross-entropy), optimizer (Adam), scheduler (Noam), and state.
- For each epoch:
  - Load train split for epoch; build iterator with `load_data_iterator`.
  - Iterate batches:
    - `model.train(); optimizer.zero_grad();`
    - Forward: `logits, extra = model(**sample['net_input'])`
    - Compute loss vs `sample['target']`; divide by `accum_steps` if accumulating.
    - `loss.backward(); clip_grad_norm_(...);`
    - On accumulation boundary: `optimizer.step(); scheduler.step(); optimizer.zero_grad();`
    - Log step/epoch/loss/ppl/lr/grad_norm.
  - Validate with `valid_loader`: `model.eval(); no_grad(); compute loss/accuracy/bleu` (bleu via sacrebleu); track best checkpoints.
  - Save checkpoints periodically; keep last `keep_last_epochs`.
- Optionally log to wandb.

## Inference / generation
- Load best checkpoint; set `model.eval()`.
- Build test iterator via `task.get_batch_iterator(split="test", ...)`.
- Use fairseq `SequenceGenerator` (beam search with `beam`, `max_len_a/b`, `post_process=sentencepiece`).
- For each batch: `task.inference_step(generator, models, sample)` to get hypotheses; detokenize via dictionary `string` with `sentencepiece` postprocess.
- Save generated Chinese sentences to output file (one per line) or CSV.

## Back-translation note
- TODO in assignment: translate target->source to create synthetic parallel data, append to train, and retrain to improve performance.

> 以上对英文版 notebook 的所有主要代码块逐段解释，涵盖数据清洗、BPE、fairseq 数据管线、RNN 编解码器与注意力、Noam 调度、训练/验证与推理。可在 TODO 处尝试 Transformer/Conformer、回译等改进以提升 BLEU。祝实验顺利！
