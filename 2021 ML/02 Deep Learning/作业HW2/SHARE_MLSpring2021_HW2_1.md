# **Homework 2-1 Phoneme Classification**

## The DARPA TIMIT Acoustic-Phonetic Continuous Speech Corpus (TIMIT)
The TIMIT corpus of reading speech has been designed to provide speech data for the acquisition of acoustic-phonetic knowledge and for the development and evaluation of automatic speech recognition systems.

This homework is a multiclass classification task, 
we are going to train a deep neural network classifier to predict the phonemes for each frame from the speech corpus TIMIT.

link: https://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3

## Download Data
Download data from google drive, then unzip it.

You should have `timit_11/train_11.npy`, `timit_11/train_label_11.npy`, and `timit_11/test_11.npy` after running this block.<br><br>
`timit_11/`
- `train_11.npy`: training data<br>
- `train_label_11.npy`: training label<br>
- `test_11.npy`:  testing data<br><br>

**notes: if the google drive link is dead, you can download the data directly from Kaggle and upload it to the workspace**





```
!gdown --id '1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR' --output data.zip
!unzip data.zip
!ls 
```

    Downloading...
    From: https://drive.google.com/uc?id=1HPkcmQmFGu-3OknddKIa5dNDsR05lIQR
    To: /content/data.zip
    372MB [00:03, 121MB/s]
    Archive:  data.zip
       creating: timit_11/
      inflating: timit_11/train_11.npy   
      inflating: timit_11/test_11.npy    
      inflating: timit_11/train_label_11.npy  
    data.zip  sample_data  timit_11


## Preparing Data
Load the training and testing data from the `.npy` file (NumPy array).


```
import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of testing data: {}'.format(test.shape))
```

    Loading data ...
    Size of training data: (1229932, 429)
    Size of testing data: (451552, 429)


## Create Dataset


```
import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None:
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

```

Split the labeled data into a training set and a validation set, you can modify the variable `VAL_RATIO` to change the ratio of validation data.


```
VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print('Size of validation set: {}'.format(val_x.shape))
```

    Size of training set: (983945, 429)
    Size of validation set: (245987, 429)


Create a data loader from the dataset, feel free to tweak the variable `BATCH_SIZE` here.


```
BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
```

Cleanup the unneeded variables to save memory.<br>

**notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables later<br>the data size is quite huge, so be aware of memory usage in colab**


```
import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect()
```




    50



## Create Model

Define model architecture, you are encouraged to change and experiment with the model architecture.


```
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39) 

        self.act_fn = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```

## Training


```
#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'
```

Fix random seeds for reproducibility.


```
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

Feel free to change the training parameters here.


```
# fix random seed for reproducibility
same_seeds(0)

# get device 
device = get_device()
print(f'DEVICE: {device}')

# training parameters
num_epoch = 20               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```


```
# start training

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward() 
        optimizer.step() 

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')

```

    [001/020] Train Acc: 0.467390 Loss: 1.812880 | Val Acc: 0.564884 loss: 1.440870
    saving model with acc 0.565
    [002/020] Train Acc: 0.594031 Loss: 1.332670 | Val Acc: 0.629594 loss: 1.209077
    saving model with acc 0.630
    [003/020] Train Acc: 0.644419 Loss: 1.154247 | Val Acc: 0.658295 loss: 1.102313
    saving model with acc 0.658
    [004/020] Train Acc: 0.672767 Loss: 1.051355 | Val Acc: 0.675568 loss: 1.040186
    saving model with acc 0.676
    [005/020] Train Acc: 0.691564 Loss: 0.982245 | Val Acc: 0.683853 loss: 1.004628
    saving model with acc 0.684
    [006/020] Train Acc: 0.705731 Loss: 0.930892 | Val Acc: 0.691707 loss: 0.977562
    saving model with acc 0.692
    [007/020] Train Acc: 0.716722 Loss: 0.890210 | Val Acc: 0.691016 loss: 0.973670
    [008/020] Train Acc: 0.726312 Loss: 0.856612 | Val Acc: 0.690207 loss: 0.971627
    [009/020] Train Acc: 0.734965 Loss: 0.827445 | Val Acc: 0.698561 loss: 0.942904
    saving model with acc 0.699
    [010/020] Train Acc: 0.741926 Loss: 0.801676 | Val Acc: 0.698854 loss: 0.946376
    saving model with acc 0.699
    [011/020] Train Acc: 0.748191 Loss: 0.779319 | Val Acc: 0.700944 loss: 0.938454
    saving model with acc 0.701
    [012/020] Train Acc: 0.754672 Loss: 0.758071 | Val Acc: 0.699423 loss: 0.940523
    [013/020] Train Acc: 0.759725 Loss: 0.739450 | Val Acc: 0.699728 loss: 0.951068
    [014/020] Train Acc: 0.765137 Loss: 0.721372 | Val Acc: 0.701903 loss: 0.938658
    saving model with acc 0.702
    [015/020] Train Acc: 0.769828 Loss: 0.704748 | Val Acc: 0.701761 loss: 0.937079
    [016/020] Train Acc: 0.774698 Loss: 0.688990 | Val Acc: 0.702293 loss: 0.938634
    saving model with acc 0.702
    [017/020] Train Acc: 0.779358 Loss: 0.674498 | Val Acc: 0.702492 loss: 0.943941
    saving model with acc 0.702
    [018/020] Train Acc: 0.783076 Loss: 0.660028 | Val Acc: 0.695195 loss: 0.966189
    [019/020] Train Acc: 0.787432 Loss: 0.646340 | Val Acc: 0.700708 loss: 0.958220
    [020/020] Train Acc: 0.791536 Loss: 0.633378 | Val Acc: 0.700643 loss: 0.957066


## Testing

Create a testing dataset, and load model from the saved checkpoint.


```
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))
```




    <All keys matched successfully>



Make prediction.


```
predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)
```

Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.


```
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
```
