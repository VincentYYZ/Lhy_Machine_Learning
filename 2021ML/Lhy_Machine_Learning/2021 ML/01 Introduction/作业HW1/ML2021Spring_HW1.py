#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import sys
import math
import argparse
from typing import Tuple, Dict, Any

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Safe for headless environments
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure
    HAS_MPL = True
except Exception:
    HAS_MPL = False


# ------------------------------
# Reproducibility
# ------------------------------
MY_SEED = 42069
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(MY_SEED)
torch.manual_seed(MY_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(MY_SEED)


# ------------------------------
# Utilities
# ------------------------------
def get_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_learning_curve(loss_record: Dict[str, list], title: str = '', out_path: str = 'learning_curve.png') -> None:
    if not HAS_MPL:
        print('[WARN] Matplotlib not available. Skipping learning curve plot.')
        return
    total_steps = len(loss_record['train'])
    if total_steps == 0 or len(loss_record['dev']) == 0:
        print('[WARN] Empty loss record. Skipping plot.')
        return
    x_1 = range(total_steps)
    step = max(1, len(loss_record['train']) // max(1, len(loss_record['dev'])))
    x_2 = x_1[::step]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    try:
        ymin = 0.0
        ymax = max(5.0, max(loss_record['train'] + loss_record['dev']))
        plt.ylim(ymin, ymax)
    except Exception:
        pass
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_pred(dv_set: DataLoader, model: nn.Module, device: str, lim: float = 35., out_path: str = 'pred_vs_gt.png', preds=None, targets=None) -> None:
    if not HAS_MPL:
        print('[WARN] Matplotlib not available. Skipping prediction scatter plot.')
        return
    if preds is None or targets is None:
        model.eval()
        preds, targets = [], []
        for x, y in dv_set:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                preds.append(pred.detach().cpu())
                targets.append(y.detach().cpu())
        preds = torch.cat(preds, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()

    figure(figsize=(5, 5))
    plt.scatter(targets, preds, c='r', alpha=0.5)
    plt.plot([-0.2, lim], [-0.2, lim], c='b')
    plt.xlim(-0.2, lim)
    plt.ylim(-0.2, lim)
    plt.xlabel('ground truth value')
    plt.ylabel('predicted value')
    plt.title('Ground Truth v.s. Prediction')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------
# Dataset & DataLoader
# ------------------------------
class COVID19Dataset(Dataset):
    ''' Dataset for loading and preprocessing the COVID19 dataset '''
    def __init__(self, path, mode='train', target_only=False):
        self.mode = mode

        # Read data into numpy arrays
        with open(path, 'r') as fp:
            data = list(csv.reader(fp))
            data = np.array(data[1:])[:, 1:].astype(float)

        if not target_only:
            feats = list(range(93))
        else:
            # Using 40 states & 2 tested_positive features (indices = 57 & 75)
            feats = list(range(40)) + [57, 75]

        if mode == 'test':
            # Testing data
            # data: 893 x 93 (40 states + day 1 (18) + day 2 (18) + day 3 (17))
            data = data[:, feats]
            self.data = torch.FloatTensor(data)
        else:
            # Training data (train/dev sets)
            # data: 2700 x 94 (40 states + day 1 (18) + day 2 (18) + day 3 (18))
            target = data[:, -1]
            data = data[:, feats]

            # Splitting training data into train & dev sets
            if mode == 'train':
                indices = [i for i in range(len(data)) if i % 10 != 0]
            elif mode == 'dev':
                indices = [i for i in range(len(data)) if i % 10 == 0]
            else:
                raise ValueError("mode must be 'train', 'dev', or 'test'")

            # Convert data into PyTorch tensors
            self.data = torch.FloatTensor(data[indices])
            self.target = torch.FloatTensor(target[indices])

        # Normalize features (you may remove this part to see what will happen)
        self.data[:, 40:] = \
            (self.data[:, 40:] - self.data[:, 40:].mean(dim=0, keepdim=True)) \
            / (self.data[:, 40:].std(dim=0, keepdim=True) + 1e-8)

        self.dim = self.data.shape[1]

        print('Finished reading the {} set of COVID19 Dataset ({} samples found, each dim = {})'
              .format(mode, len(self.data), self.dim))

    def __getitem__(self, index):
        # Returns one sample at a time
        if self.mode in ['train', 'dev']:
            # For training
            return self.data[index], self.target[index]
        else:
            # For testing (no target)
            return self.data[index]

    def __len__(self):
        # Returns the size of the dataset
        return len(self.data)


def prep_dataloader(path, mode, batch_size, n_jobs=0, target_only=False):
    ''' Generates a dataset, then is put into a dataloader. '''
    dataset = COVID19Dataset(path, mode=mode, target_only=target_only)  # Construct dataset
    dataloader = DataLoader(
        dataset, batch_size,
        shuffle=(mode == 'train'), drop_last=False,
        num_workers=n_jobs, pin_memory=True)                            # Construct dataloader
    return dataloader


# ------------------------------
# Model
# ------------------------------
class NeuralNet(nn.Module):
    ''' A simple fully-connected deep neural network '''
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()

        # Define your neural network here
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Mean squared error loss
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, x):
        ''' Given input of size (batch_size x input_dim), compute output of the network '''
        return self.net(x).squeeze(1)

    def cal_loss(self, pred, target):
        ''' Calculate loss '''
        return self.criterion(pred, target)


# ------------------------------
# Train / Dev / Test
# ------------------------------
def train(tr_set, dv_set, model, config, device):
    ''' DNN training '''

    n_epochs = config['n_epochs']  # Maximum number of epochs

    # Setup optimizer
    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    min_mse = 1000.
    loss_record = {'train': [], 'dev': []}      # for recording training loss
    early_stop_cnt = 0
    epoch = 0
    while epoch < n_epochs:
        model.train()                           # set model to training mode
        for x, y in tr_set:                     # iterate through the dataloader
            optimizer.zero_grad()               # set gradient to zero
            x, y = x.to(device), y.to(device)   # move data to device (cpu/cuda)
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
            mse_loss.backward()                 # compute gradient (backpropagation)
            optimizer.step()                    # update model with optimizer
            loss_record['train'].append(mse_loss.detach().cpu().item())

        # After each epoch, test your model on the validation (development) set.
        dev_mse = dev(dv_set, model, device)
        if dev_mse < min_mse:
            # Save model if your model improved
            min_mse = dev_mse
            print('Saving model (epoch = {:4d}, loss = {:.4f})'
                .format(epoch + 1, min_mse))
            torch.save(model.state_dict(), config['save_path'])  # Save model to specified path
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        epoch += 1
        loss_record['dev'].append(dev_mse)
        if early_stop_cnt > config['early_stop']:
            # Stop training if your model stops improving for "config['early_stop']" epochs.
            break

    print('Finished training after {} epochs'.format(epoch))
    return min_mse, loss_record


def dev(dv_set, model, device):
    model.eval()                                # set model to evalutation mode
    total_loss = 0
    for x, y in dv_set:                         # iterate through the dataloader
        x, y = x.to(device), y.to(device)       # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            mse_loss = model.cal_loss(pred, y)  # compute loss
        total_loss += mse_loss.detach().cpu().item() * len(x)  # accumulate loss
    total_loss = total_loss / len(dv_set.dataset)              # compute averaged loss

    return total_loss


def test(tt_set, model, device):
    model.eval()                                # set model to evalutation mode
    preds = []
    for x in tt_set:                            # iterate through the dataloader
        x = x.to(device)                        # move data to device (cpu/cuda)
        with torch.no_grad():                   # disable gradient calculation
            pred = model(x)                     # forward pass (compute output)
            preds.append(pred.detach().cpu())   # collect prediction
    preds = torch.cat(preds, dim=0).numpy()     # concatenate all predictions and convert to a numpy array
    return preds


def save_pred(preds: np.ndarray, file: str) -> None:
    ''' Save predictions to specified file '''
    print('Saving results to {}'.format(file))
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])


# ------------------------------
# Data download helper (optional)
# ------------------------------
DRIVE_IDS = {
    'train': '19CCyCgJrUxtvgZF53vnctJiOJ23T5mqF',
    'test': '1CE240jLm2npU-tdz81-oVKEF3T2yfT1O',
}

def try_download_with_gdown(out_path: str, file_id: str) -> bool:
    try:
        import gdown  # type: ignore
    except Exception:
        print('[WARN] gdown is not installed; skip downloading {}.'.format(out_path))
        return False
    try:
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f'Downloading {out_path} from {url} ...')
        gdown.download(url, out_path, quiet=False)
        return os.path.exists(out_path)
    except Exception as e:
        print(f'[WARN] gdown download failed for {out_path}: {e}')
        return False


# ------------------------------
# Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='ML2021 Spring HW1 - COVID-19 Regression (PyTorch)')
    parser.add_argument('--train-path', type=str, default='covid.train.csv')
    parser.add_argument('--test-path', type=str, default='covid.test.csv')
    parser.add_argument('--download', action='store_true', help='Download dataset via gdown if missing')
    parser.add_argument('--target-only', action='store_true', help='Use 40 states + 2 tested_positive features')  
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=270)
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--early-stop', type=int, default=200)
    parser.add_argument('--save-path', type=str, default=os.path.join('models', 'model.pth'))
    parser.add_argument('--num-workers', type=int, default=0)

    parser.add_argument('--plot', action='store_true', help='Save learning curve and pred plots')
    parser.add_argument('--pred-out', type=str, default='pred.csv')

    args = parser.parse_args()

    device = get_device()
    print('[Info] Using device:', device)

    # Ensure model dir exists
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)

    # Optionally download datasets
    if args.download:
        if not os.path.exists(args.train_path):
            ok = try_download_with_gdown(args.train_path, DRIVE_IDS['train'])
            if not ok:
                print('[WARN] Failed to download train data. Please place file at:', args.train_path)
        if not os.path.exists(args.test_path):
            ok = try_download_with_gdown(args.test_path, DRIVE_IDS['test'])
            if not ok:
                print('[WARN] Failed to download test data. Please place file at:', args.test_path)

    # Prepare data loaders
    if not os.path.exists(args.train_path) or not os.path.exists(args.test_path):
        print('[ERROR] Data files not found. Provide --train-path/--test-path or use --download.')
        sys.exit(1)

    tr_set = prep_dataloader(args.train_path, 'train', args.batch_size, n_jobs=args.num_workers, target_only=args.target_only)
    dv_set = prep_dataloader(args.train_path, 'dev', args.batch_size, n_jobs=args.num_workers, target_only=args.target_only)
    tt_set = prep_dataloader(args.test_path, 'test', args.batch_size, n_jobs=args.num_workers, target_only=args.target_only)

    # Build model
    model = NeuralNet(tr_set.dataset.dim).to(device)

    # Train
    config = {
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'optimizer': args.optimizer,
        'optim_hparas': {
            'lr': args.lr,
            **({'momentum': args.momentum} if args.optimizer == 'SGD' else {}),
        },
        'early_stop': args.early_stop,
        'save_path': args.save_path,
    }

    best_mse, loss_record = train(tr_set, dv_set, model, config, device)

    if args.plot:
        plot_learning_curve(loss_record, title='deep model', out_path='learning_curve.png')

    # Load best and show pred plot (optional)
    del model
    model = NeuralNet(tr_set.dataset.dim).to(device)
    ckpt = torch.load(config['save_path'], map_location=device)
    model.load_state_dict(ckpt)

    if args.plot:
        plot_pred(dv_set, model, device, out_path='pred_vs_gt.png')

    # Test and save predictions
    preds = test(tt_set, model, device)
    save_pred(preds, args.pred_out)


if __name__ == '__main__':
    main() 