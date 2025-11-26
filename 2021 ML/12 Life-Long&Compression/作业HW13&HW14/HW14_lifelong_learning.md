### LifeLong Machine Learning
### TA's Slide
[Slide](https://docs.google.com/presentation/d/13JmcOZ9i_m5xJbRBKNMAKE1fIzGhyaeLck3frY0B2xY/edit?usp=sharing)

### Definition
The detailed explanations and definitions of LifeLong Learning please refer to [LifeLong learning](https://youtu.be/7qT5P9KJnWo) 


### Methods
Someone proposed a survey paper for LifeLong Learning at the end of 2019 to distinguish 2016-2019 LigeLong Learning methods into three families.

We can distinguish LifeLong Learning methods into three families, based on how task
specific information is stored and used throughout the sequential learning process:
* Replay-based methods
* Regularization-based methods
* Parameter isolation methods

<img src="https://i.ibb.co/VDFJkWG/2019-12-29-17-25.png" width="100%">

In this assignment, we have to go through EWC, MAS, SI, Remanian Walk, SCP Methods in the prior-focused methods of the regularization-based methods. 

Source: [Continual Learning in Neural
Networks](https://arxiv.org/pdf/1910.02718.pdf)

Please feel free to mail us if you have any questions.

ntu-ml-2020spring-ta@googlegroups.com



### Table of Content
- Utils
- Visualization
- Methods

### Utility
We utilize permuted MNIST as our training dataset.

So, first we utilize 5 different permutations to generate 10 different permuted MNIST as different task.

 #### - Pemutation


```
import torch.utils.data as data
import torch.utils.data.sampler as sampler
import torchvision
import os
import torch.nn.functional as F
from torchvision import datasets, transforms

# Permute MNIST to generate 10 tasks

def _permutate_image_pixels(image, permutation):
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    image.view(c, h, w)
    return image

def get_transform(permutation=None, normalize=True):
  if normalize == True:
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(28),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])
  else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(28),
                                    transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
    ])
  return transform

class Pad(object):
  def __init__(self, size, fill=0, padding_mode='constant'):
    self.size = size
    self.fill = fill
    self.padding_mode = padding_mode
    
  def __call__(self, img):
    # If the H and W of img is not equal to desired size,
    # then pad the channel of img to desired size.
    img_size = img.size()[1]
    assert ((self.size - img_size) % 2 == 0)
    padding = (self.size - img_size) // 2
    padding = (padding, padding, padding, padding)
    return F.pad(img, padding, self.padding_mode, self.fill)

class Data():
  def __init__(self, path, train=True, permutation=None, normalize=True):

    transform = get_transform(permutation, normalize)
    self.dataset = datasets.MNIST(root = os.path.join(path, "MNIST"),
                                        transform=transform,
                                        train = train,
                                        download = True)
```

#### - Dataloader and Argument
- Training Arguments
- Setup 5 different Permutation
- 5 Train DataLoader
- 5 Test DataLoader 



```
### Main Process

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader
from tqdm import trange

class Args:
  task_number = 5
  epochs_per_task = 10
  lr = 1.0e-4
  batch_size = 128
  test_size=8192
  random_seed=0

args=Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# generate permutations for the tasks.
np.random.seed(args.random_seed)

#generate permuted MNIST data from 10 different permutation.
permutations = [
    np.random.permutation(784) if index !=0 else np.arange(784) for index in range(args.task_number) ]

# prepare permuted mnist datasets.
train_datasets = [
    Data('data', permutation=permutations[index]) for index in range(len(permutations))
]
train_dataloaders = [
    DataLoader(data.dataset, batch_size=args.batch_size, shuffle=True) for data in train_datasets
]


test_datasets = [
    Data('data',train=False, permutation=permutations[index]) for index in range(len(permutations))
]
test_dataloaders = [
    DataLoader(data.dataset, batch_size=args.test_size, shuffle=True) for data in test_datasets
]

```

    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz



    HBox(children=(FloatProgress(value=0.0, max=9912422.0), HTML(value='')))


    
    Extracting data/MNIST/MNIST/raw/train-images-idx3-ubyte.gz to data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz



    HBox(children=(FloatProgress(value=0.0, max=28881.0), HTML(value='')))


    
    Extracting data/MNIST/MNIST/raw/train-labels-idx1-ubyte.gz to data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
    Failed to download (trying next):
    HTTP Error 503: Service Unavailable
    
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
    Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz



    HBox(children=(FloatProgress(value=0.0, max=1648877.0), HTML(value='')))


    
    Extracting data/MNIST/MNIST/raw/t10k-images-idx3-ubyte.gz to data/MNIST/MNIST/raw
    
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
    Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz to data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz



    HBox(children=(FloatProgress(value=0.0, max=4542.0), HTML(value='')))


    
    Extracting data/MNIST/MNIST/raw/t10k-labels-idx1-ubyte.gz to data/MNIST/MNIST/raw
    
    Processing...
    Done!


    /usr/local/lib/python3.7/dist-packages/torchvision/datasets/mnist.py:502: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:143.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)


#### - Model
To fair comparison, 

We fix our model architecture to do this homework. 

The model architecture consist 4 layers fully-connected network.


```
import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
  """
  Model architecture 
  784 (input) → 1024 → 512 → 256 → 10
  """
  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = nn.Linear(784, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 256)
    self.fc4 = nn.Linear(256, 10)
    self.relu = nn.ReLU()

  def forward(self, x):
    x = x.view(-1, 1*28*28)
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    x = self.fc4(x)
    return x

example = Model()
print(example)

```

    Model(
      (fc1): Linear(in_features=784, out_features=1024, bias=True)
      (fc2): Linear(in_features=1024, out_features=512, bias=True)
      (fc3): Linear(in_features=512, out_features=256, bias=True)
      (fc4): Linear(in_features=256, out_features=10, bias=True)
      (relu): ReLU()
    )


#### - Train
This is our function of training process.

It can generally applied in different regularization-based lifelong learning algorithm in this homework.


```
import torch
import torch.nn as nn
import tqdm
import numpy as np
from tqdm import trange

def train(model, optimizer, dataloader, epochs_per_task, lll_object, lll_lambda, test_dataloaders, evaluate, device, log_step=1):
    model.train()
    model.zero_grad()
    objective = nn.CrossEntropyLoss()
    acc_per_epoch = []
    loss = 1.0
    bar = tqdm.auto.trange(epochs_per_task, leave=False, desc=f"Epoch 1, Loss: {loss:.7f}")
    for epoch in bar:
        for imgs, labels in tqdm.auto.tqdm(dataloader, leave=False):            
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = objective(outputs, labels)
            total_loss = loss
            lll_loss = lll_object.penalty(model)
            total_loss += lll_lambda * lll_loss 
            lll_object.update(model)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.item()
            bar.set_description_str(desc=f"Epoch {epoch+1:2}, Loss: {loss:.7f}", refresh=True)
        acc_average  = []
        for test_dataloader in test_dataloaders: 
            acc_test = evaluate(model, test_dataloader, device)
            acc_average.append(acc_test)
        average=np.mean(np.array(acc_average))
        acc_per_epoch.append(average*100.0)
        bar.set_description_str(desc=f"Epoch {epoch+2:2}, Loss: {loss:.7f}", refresh=True)
                
    return model, optimizer, acc_per_epoch
```

#### - Evaluate
This is our function of evaluation process.

It can generally applied in different regularization-based lifelong learning algorithm in this homework.


```
import torch
import torch.nn as nn


def evaluate(model, test_dataloader, device):
    model.eval()
    correct_cnt = 0
    total = 0
    for imgs, labels in test_dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, pred_label = torch.max(outputs.data, 1)

        correct_cnt += (pred_label == labels.data).sum().item()
        total += torch.ones_like(labels.data).sum().item()
    return correct_cnt / total
```

#### - Evaluation Metric
We utilize **Average Accuracy** as our evaluation metric, 

which average the accuracy from the all previous and current test set to measure the performance of lifelong learning . 

### Visualization


#### - Permuted MNIST dataset


```
### Visualize label 0-9 1 sample MNIST picture in first 3 task.

sample = [
    Data('data', permutation=permutations[index], normalize=False) for index in range(len(permutations))
]
import matplotlib.pyplot as plt
plt.figure(figsize=(30, 10))
for task in range(3):
  labels = [list(sample[task].dataset.targets).index(l) for l in range(10)]
  for idx, label in enumerate(labels):
    plt.subplot(3, 10, (task)*10 + idx + 1)
    curr_img = np.reshape(sample[task].dataset[label][0], (28, 28))
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'), fignum=False)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("task: " + str(task+1) + " " + "label: " + str(idx), y=1)


```


    
![png](HW14_lifelong_learning_files/HW14_lifelong_learning_16_0.png)
    


### Methods
- Baseline
- EWC
- SI
- MAS
- RWalk
-SCP

#### - Baseline
The baseline class will do nothing in regularization term.


```
# Baseline 
import torch
import torch.nn as nn


class baseline(object):
    """
    baseline technique: do nothing in regularization term [initialize and all weight is zero]
    """
    def __init__(self, model, dataloaders, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
        self.p_old = {} # store current parameters
        self._precision_matrices = self._calculate_importance() # generate weight matrix 

        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
  
    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items(): # initialize weight matrix（fill zero）
            precision_matrices[n] = p.clone().detach().fill_(0)

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        # do nothing
        return 

```

Main process for baseline


```
# Baseline
print("RUN BASELINE")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# initialize lifelong learning object (baseline class) without adding any regularization term.
lll_object=baseline(model=model, dataloaders=[None],device=device)
lll_lambda=0.0
baseline_acc= []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

# iterate training on each task continually.
for train_indexes in task_bar:
    # Train each task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])

    # get model weight to baseline class and do nothing!
    lll_object=baseline(model=model, dataloaders=test_dataloaders[:train_indexes],device=device)

    # new a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    baseline_acc.extend(acc_list)
    
    # display the information of the next task.
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch! 
print(baseline_acc)
print("==================================================================================================")

```

    RUN BASELINE



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.97, 96.09, 96.58, 97.34, 97.63, 97.82, 97.86, 98.11, 98.04, 97.99, 96.05999999999999, 96.71499999999999, 96.69000000000001, 96.695, 96.52499999999999, 96.21, 96.24999999999999, 96.475, 95.93, 95.98, 94.40333333333334, 93.85, 92.73666666666665, 91.24666666666667, 91.22666666666667, 90.93333333333332, 90.83333333333333, 90.05666666666666, 89.72333333333333, 88.21, 88.9125, 88.6, 87.995, 87.02499999999999, 86.195, 86.34249999999999, 86.6175, 85.87750000000001, 86.2975, 84.645, 86.394, 85.306, 85.71, 85.26599999999999, 85.348, 85.694, 85.552, 85.47399999999999, 84.42799999999998, 83.72000000000001]
    ==================================================================================================


#### - EWC

Elastic Weight Consolidation

The ewc class applied EWC algorithm to calculate regularization term.
The central concept is included in Prof.Hung-yi's lectures. Here we will focus on the algorithm of EWC. 

In this assignment, we want to let our model learn 10 tasks successively.
Here we show a simple example that we let the model learn 2 tasks(task A, task B) successively.

In EWC algorithm, the definition of loss function is shown below:
 $$\mathcal{L}_B = \mathcal{L}(\theta) + \sum_{i} \frac{\lambda}{2} F_i (\theta_{i} - \theta_{A,i}^{*})^2  $$
  
Assume we have a neural network with more than two parameters.

$F_i$ correspond to the $i^{th}$ guard in Prof. Hung-yi's lecture. Please do not modify this parameters, because it's important to task A.

The definition of $F$ is shown below.
$$ F = [ \nabla \log(p(y_n | x_n, \theta_{A}^{*}) \nabla \log(p(y_n | x_n, \theta_{A}^{*})^T ] $$ 

We only take the diagonal value of matrix to approximate each parameters' $F_i$.

The detail infromation and derivation are shown in 2.4.1 and 2.4 of [Continual Learning in Neural
Networks](https://arxiv.org/pdf/1910.02718.pdf)

For You Information: [Elastic Weight Consolidation](https://arxiv.org/pdf/1612.00796.pdf)



```
import torch
import torch.nn as nn
import torch.nn.functional as F

class ewc(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
  """
    def __init__(self, model, dataloaders, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device

        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract all parameters in models
        self.p_old = {} # initialize parameters
        self._precision_matrices = self._calculate_importance() # generate Fisher (F) matrix for EWC 

        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
  
    def _calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items(): 
            # initialize Fisher (F) matrix（all fill zero）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num=len(self.dataloaders)
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    # get image data
                    input = data[0].to(self.device)
                    # image data forward model
                    output = self.model(input)
                    # Simply use groud truth label of dataset.  
                    label = data[1].to(self.device)
                    # print(output.shape, label.shape)
                    
                    ############################################################################
                    #####                     generate Fisher(F) matrix for EWC            #####
                    ############################################################################    
                    loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
                    loss.backward()                                                    
                    ############################################################################

                    for n, p in self.model.named_parameters():
                        # get the gradient of each parameter and square it, then average it in all validation set.                          
                        precision_matrices[n].data += p.grad.data ** 2 / number_data   
                                                                            
            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            # generate the final regularization term by the ewc weight (self._precision_matrices[n]) and the square of weight difference ((p - self.p_old[n]) ** 2).  
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        # do nothing
        return 

```

Main process for EWC 


```
#EWC
print("RUN EWC")
model = Model()
model = model.to(device)
# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# initialize lifelong learning object for EWC
lll_object=ewc(model=model, dataloaders=[None],device=device)

# setup the coefficient value of regularization term.
lll_lambda=100
ewc_acc= []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")

# iterate training on each task continually.
for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    
    # get model weight and calculate guidance for each weight
    lll_object=ewc(model=model, dataloaders=test_dataloaders[:train_indexes+1],device=device)

    # new a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # collect average accuracy in each epoch
    ewc_acc.extend(acc_list)

    # Update tqdm displayer
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch!     
print(ewc_acc)
print("==================================================================================================")


```

    RUN EWC



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.89, 95.95, 96.77, 97.04, 97.5, 97.59, 97.82, 97.95, 98.04, 97.99, 96.22500000000001, 96.88000000000001, 96.59500000000001, 96.77, 97.12, 97.06, 97.255, 97.025, 97.15, 97.15, 95.71, 95.75333333333334, 95.47333333333334, 95.25333333333333, 95.16666666666667, 94.97999999999999, 94.89000000000001, 94.92333333333333, 94.71, 94.24666666666667, 93.57249999999999, 93.38250000000001, 92.6275, 92.695, 92.225, 92.82000000000001, 92.81750000000001, 92.3875, 92.255, 92.44999999999999, 91.75999999999999, 91.364, 91.372, 91.754, 91.368, 91.788, 91.674, 91.588, 91.536, 91.09400000000001]
    ==================================================================================================


#### - MAS
Memory Aware Synapses

The mas class applied MAS algorithm to calculate regularization term.

The concept of MAS is similar to EWC, the only difference is the calculation of the important weight. 
The details are mentioned in following blocks.

MAS:

In MAS, the Loss function is shown below, the model learn task A before it learned task B.

$$\mathcal{L}_B = \mathcal{L}(\theta) + \sum_{i} \frac{\lambda}{2} \Omega_i (\theta_{i} - \theta_{A,i}^{*})^2$$

Compare with EWC, the $F_i$ in the loss function is replaced with $\Omega_i$ in the following function.

$$\Omega_i = || \frac{\partial \ell_2^2(M(x_k; \theta))}{\partial \theta_i} || $$ 

$x_k$ is the sample data of the previous task. So the $\Omega$ is obtained gradients of the squared L2-norm of the learned network output.

The methods that proposed from paper is the local version by taking squared L2-norm outputs from the each layers of the model.

Here we only implmented the global version by taking outputs from the last layer of the model. 


For Your Information: 
[Memory Aware Synapses](https://arxiv.org/pdf/1711.09601.pdf)
 




```
class mas(object):
    """
    @article{aljundi2017memory,
      title={Memory Aware Synapses: Learning what (not) to forget},
      author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
      booktitle={ECCV},
      year={2018},
      url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """
    def __init__(self, model: nn.Module, dataloaders: list, device):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #extract all parameters in models
        self.p_old = {} # initialize parameters
        self.device = device
        self._precision_matrices = self.calculate_importance() # generate Omega(Ω) matrix for MAS
    
        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach() # keep the old parameter in self.p_old
    
    def calculate_importance(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0) # initialize Omega(Ω) matrix（all filled zero）

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num = len(self.dataloaders)
            num_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    output = self.model(data[0].to(self.device))

                    ###########################################################################################################################################
                    #####  TODO BLOCK: generate Omega(Ω) matrix for MAS. (Hint: square of l2 norm of output vector, then backward and take its gradients  #####
                    ###########################################################################################################################################
                    output.pow_(2)                                                   
                    loss = torch.sum(output,dim=1)                                   
                    loss = loss.mean()   
                    loss.backward() 
                    ###########################################################################################################################################                          
                                            
                    for n, p in self.model.named_parameters():                      
                        precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      
                        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        # do nothing
        return 
```

Main process for MAS


```
    # MAS
    print("RUN MAS")
    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lll_object=mas(model=model, dataloaders=[None],device=device)
    lll_lambda=0.1
    mas_acc= []
    task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
    for train_indexes in task_bar:
        # Train Each Task
        model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
        
        # get model weight and calculate guidance for each weight
        lll_object=mas(model=model, dataloaders=test_dataloaders[:train_indexes+1],device=device)

        # New a Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Collect average accuracy in each epoch
        mas_acc.extend(acc_list)
        task_bar.set_description_str(f"Task  {train_indexes+2:2}")
    
    # average accuracy in each task per epoch!     
    print(mas_acc)
    print("==================================================================================================")

```

    RUN MAS



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.87, 96.02000000000001, 96.71, 97.09, 97.53, 97.72, 97.76, 97.84, 98.05, 98.0, 95.845, 96.28999999999999, 96.685, 96.58, 96.61, 97.00999999999999, 97.005, 96.975, 96.825, 97.02, 95.52333333333335, 96.12666666666667, 96.01666666666667, 96.20333333333333, 96.21666666666667, 96.11999999999999, 96.41666666666667, 96.22, 96.47666666666666, 96.15, 95.13000000000001, 95.4675, 95.76249999999999, 95.795, 95.52000000000001, 95.66749999999999, 96.00750000000001, 95.83749999999999, 95.5025, 95.74499999999999, 94.696, 94.58400000000002, 94.804, 94.85, 94.894, 94.91999999999999, 95.16, 95.144, 94.952, 95.094]
    ==================================================================================================


#### - SI
The si class applied SI (Synaptic Intelligence) algorithm to calculate regularization term.


```
import torch
import torch.nn as nn
import torch.nn.functional as F


class si(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
  """
    def __init__(self, model, dataloaders, epsilon, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.epsilon = epsilon
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} #抓出模型的所有參數
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()

    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        if self.dataloaders[0] != None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W/(p_change**2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current


                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)
        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())


        return n_p_prev, n_omega

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                _loss = omega * (p - prev_values) ** 2
                loss += _loss.sum()
         
        return loss
    
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return 
 

```

Main Process for SI



```
    # SI
    print("RUN SI")
    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lll_object=si(model=model, dataloaders=[None], epsilon=0.1, device=device)
    lll_lambda=1
    si_acc = []
    task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
    for train_indexes in task_bar:
        # Train Each Task
        model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
        
        # get model weight and calculate guidance for each weight
        lll_object=si(model=model, dataloaders=test_dataloaders[:train_indexes+1], epsilon=0.1, device=device)

        # New a Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Collect average accuracy in each epoch
        si_acc.extend(acc_list)
        task_bar.set_description_str(f"Task  {train_indexes+2:2}")

    # average accuracy in each task per epoch!     
    print(si_acc)
```

    RUN SI



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.65, 96.12, 96.7, 97.3, 97.37, 97.96000000000001, 97.41, 97.86, 97.97, 98.16, 96.55, 97.18, 97.095, 97.18500000000002, 97.165, 97.1, 97.22500000000001, 96.93499999999999, 97.0, 97.31, 95.90333333333334, 96.06333333333333, 95.71666666666665, 95.11666666666666, 95.73, 95.33333333333333, 95.39666666666666, 95.13666666666666, 95.28666666666666, 94.90333333333334, 93.1575, 92.99499999999999, 92.0825, 92.14750000000001, 91.7425, 91.6175, 92.4725, 92.015, 91.41999999999999, 92.58, 91.05000000000001, 90.41799999999999, 90.25399999999999, 90.43599999999999, 90.28399999999999, 89.524, 90.168, 90.66000000000001, 89.87399999999998, 90.428]


#### - RWalk

#### Remanian Walk for Incremental Learning

The rwalk class applied Remanian Walk algorithm to calculate regularization term.

The details are mentioned in following blocks.


```
import torch
import torch.nn as nn
import torch.nn.functional as F

class rwalk(object):
    """

    """
    def __init__(self, model, dataloaders, epsilon, device):
    
        self.model = model
        self.dataloaders = dataloaders
        self.device = device
        self.epsilon = epsilon
        self.update_ewc_parameter = 0.4
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad} # extract model parameters and store in dictionary
        self._means = {} # initialize the guidance matrix
        self._precision_matrices = self._calculate_importance_ewc() # Generate Fisher (F) Information Matrix 
        self._n_p_prev, self._n_omega = self._calculate_importance() 
        self.W, self.p_old = self._init_()


    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        if self.dataloaders[0] != None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:

                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W / (1.0 / 2.0*self._precision_matrices[n] *p_change**2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = 0.5 * omega + 0.5 *omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current


                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)
        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())


        return n_p_prev, n_omega
    

    def _calculate_importance_ewc(self):
        precision_matrices = {}
        for n, p in self.params.items(): 
            n = n.replace('.', '__') # 初始化 Fisher (F) 的矩陣（都補零）
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num=len(self.dataloaders)
            number_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for n, p in self.model.named_parameters():                         
                    n = n.replace('.', '__')
                    precision_matrices[n].data *= (1 -self.update_ewc_parameter)   
                for data in dataloader:
                    self.model.zero_grad()
                    input = data[0].to(self.device)
                    output = self.model(input)
                    label = data[1].to(self.device)

                    
                    ############################################################################
                    #####                      Generate Fisher Matrix                      #####
                    ############################################################################    
                    loss = F.nll_loss(F.log_softmax(output, dim=1), label)             
                    loss.backward()                                                    
                                                                                    
                    for n, p in self.model.named_parameters():                         
                        n = n.replace('.', '__')
                        precision_matrices[n].data += self.update_ewc_parameter*p.grad.data ** 2 / number_data  
                                                                            
            precision_matrices = {n: p for n, p in precision_matrices.items()}

        return precision_matrices


    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]

                #################################################################################
                ####        Generate regularization term  _loss by omega and Fisher Matrix   ####
                #################################################################################
                _loss = (omega + self._precision_matrices[n]) * (p - prev_values) ** 2
                loss += _loss.sum()
         
        return loss
    
    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return 
 

```

Main Process for RWalk


```
# RWalk
print("RUN Rwalk")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=rwalk(model=model, dataloaders=[None], epsilon=0.1, device=device)
lll_lambda=100
rwalk_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
for train_indexes in task_bar:
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    lll_object=rwalk(model=model, dataloaders=test_dataloaders[:train_indexes+1], epsilon=0.1, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rwalk_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch!     
print(rwalk_acc)
print("==================================================================================================")

```

    RUN Rwalk



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.36, 95.66, 96.74000000000001, 96.95, 97.57000000000001, 97.75, 97.7, 98.00999999999999, 97.76, 97.89999999999999, 96.06, 97.03500000000001, 96.9, 97.275, 97.16, 97.21, 97.38, 97.255, 97.39999999999999, 97.32, 95.37666666666667, 96.08666666666666, 96.07333333333334, 96.30666666666666, 96.31333333333333, 96.22333333333333, 96.36333333333334, 96.08666666666666, 95.90666666666667, 96.07666666666667, 94.65, 95.075, 94.9325, 95.06500000000001, 94.80499999999999, 95.0975, 95.2225, 95.1975, 94.9925, 94.78, 93.96000000000001, 94.214, 93.836, 94.036, 94.152, 94.00800000000001, 94.17, 94.178, 93.89800000000001, 93.986]
    ==================================================================================================


#### - SCP
Sliced Cramer Preservation



Pseudo Code:

<img src="https://i.ibb.co/QJycmNZ/2021-02-18-21-07.png" width="100%">



```
import torch
from torch import nn
import numpy as np



def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return torch.from_numpy(vec)

class scp(object):
    """
    OPEN REVIEW VERSION:
    https://openreview.net/forum?id=BJge3TNKwH
    """
    def __init__(self, model: nn.Module, dataloaders: list, L: int, device):
        self.model = model 
        self.dataloaders = dataloaders
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._state_parameters = {}
        self.L= L
        self.device = device
        self._precision_matrices = self.calculate_importance()
    
        for n, p in self.params.items():
            self._state_parameters[n] = p.clone().detach()
    
    def calculate_importance(self):

        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        self.model.eval()
        if self.dataloaders[0] is not None:
            dataloader_num = len(self.dataloaders)
            num_data = sum([len(loader) for loader in self.dataloaders])
            for dataloader in self.dataloaders:
                for data in dataloader:
                    self.model.zero_grad()
                    output = self.model(data[0].to(self.device))
                    
                    ####################################################################################
                    ##### generate SCP's Gamma(Γ) matrix (like MAS's Omega(Ω) and EWC's Fisher(F)) #####
                    ####################################################################################
                    #####        1.take average on a batch of Output vector to get vector φ(:,θ_A* )####
                    ####################################################################################
                    mean_vec = output.mean(dim=0)

                    ####################################################################################
                    #####   2. random sample L vectors ξ #（ Hint: sample_spherical() ）      #####
                    ####################################################################################
                    L_vectors = sample_spherical(self.L, output.shape[-1])
                    L_vectors = L_vectors.transpose(1,0).to(self.device).float()

                    ####################################################################################
                    #####   3.    每一個 vector ξ 和 vector φ( :,θ_A* )內積得到 scalar ρ               ####
                    #####           對 scalar ρ 取 backward ， 每個參數得到各自的 gradient ∇ρ           ####
                    #####       每個參數的 gradient ∇ρ 取平方 取 L 平均 得到 各個參數的 Γ scalar          ####  
                    #####              所有參數的  Γ scalar 組合而成其實就是 Γ 矩陣                      ####
                    ####(hint: 記得 每次 backward 之後 要 zero_grad 去 清 gradient, 不然 gradient會累加 )####   
                    ####################################################################################
                    total_scalar = 0
                    for vec in L_vectors:
                        scalar=torch.matmul(vec, mean_vec)
                        total_scalar += scalar
                    total_scalar /= L_vectors.shape[0] 
                    total_scalar.backward()
                    ##################################################################################      
                     
                                                
                    for n, p in self.model.named_parameters():                      
                        precision_matrices[n].data += p.grad.abs() / num_data ## difference with EWC      
                        
        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._state_parameters[n]) ** 2
            loss += _loss.sum()
        return loss
    
    def update(self, model):
        # do nothing
        return 
```

Main process for SCP


```
# SCP
print("RUN SLICE CRAMER PRESERVATION")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=scp(model=model, dataloaders=[None], L=100, device=device)
lll_lambda=100
scp_acc= []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
for train_indexes in task_bar:
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
    lll_object=scp(model=model, dataloaders=test_dataloaders[:train_indexes+1], L=100, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scp_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch!     
print(scp_acc)
print("==================================================================================================")

```

    RUN SLICE CRAMER PRESERVATION



    HBox(children=(FloatProgress(value=0.0, description='Task   1', max=5.0, style=ProgressStyle(description_width…



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, description='Epoch 1, Loss: 1.0000000', max=10.0, style=ProgressStyle(…



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    
    [93.75, 96.1, 96.78, 97.28, 97.78, 97.28, 97.81, 97.96000000000001, 97.89999999999999, 98.21, 95.30499999999999, 95.755, 95.89, 96.185, 96.095, 96.12, 95.98, 96.39, 96.06, 96.065, 94.58666666666667, 94.58000000000001, 94.62333333333333, 94.25333333333333, 94.26666666666668, 94.24333333333334, 93.56333333333335, 93.32666666666665, 93.25000000000001, 92.38666666666667, 92.6875, 92.16250000000001, 92.73500000000001, 92.31, 92.865, 92.225, 92.74499999999999, 92.40249999999999, 92.61500000000001, 92.245, 91.668, 92.46000000000001, 92.22999999999999, 92.712, 92.608, 92.722, 92.31399999999998, 92.462, 92.78800000000001, 92.624]
    ==================================================================================================

