Homework 13 - Network Compression
===

> Author: Arvin Liu (r09922071@ntu.edu.tw), this colab is modified from ML2021-HW3

If you have any questions, feel free to ask: ntu-ml-2021spring-ta@googlegroups.com

## **Intro**

HW13 is about network compression

There are many types of Network/Model Compression,  here we introduce two:
* Knowledge Distillation
* Design Architecture


The process of this notebook is as follows: <br/>
1. Introduce depthwise, pointwise and group convolution in MobileNet.
2. Design the model of this colab
3. Introduce Knowledge-Distillation
4. Set up TeacherNet and it would be helpful in training


## **About the Dataset**  *(same as HW3)*

The dataset used here is food-11, a collection of food images in 11 classes.

For the requirement in the homework, TAs slightly modified the data.
Please DO NOT access the original fully-labeled training data or testing labels.

Also, the modified dataset is for this course only, and any further distribution or commercial use is forbidden.


```
### This block is same as HW3 ###
# Download the dataset
# You may choose where to download the data.

# Google Drive
!gdown --id '1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy' --output food-11.zip
# If you cannot successfully gdown, you can change a link. (Backup link is provided at the bottom of this colab tutorial).

# Dropbox
# !wget https://www.dropbox.com/s/m9q6273jl3djall/food-11.zip -O food-11.zip

# MEGA
# !sudo apt install megatools
# !megadl "https://mega.nz/#!zt1TTIhK!ZuMbg5ZjGWzWX1I6nEUbfjMZgCmAgeqJlwDkqdIryfg"

# Unzip the dataset.
# This may take some time.
!unzip -q food-11.zip
```

    Downloading...
    From: https://drive.google.com/uc?id=1awF7pZ9Dz7X1jn1_QAiKN-_v56veCEKy
    To: /content/food-11.zip
    963MB [00:14, 67.3MB/s]
    replace food-11/training/unlabeled/00/5176.jpg? [y]es, [n]o, [A]ll, [N]one, [r]ename: N


## **Import Packages**  *(same as HW3)*

First, we need to import packages that will be used later.

In this homework, we highly rely on **torchvision**, a library of PyTorch.


```
### This block is same as HW3 ###
# Import necessary packages.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision.models as models

from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision.datasets import DatasetFolder

# This is for the progress bar.
from tqdm.auto import tqdm
```

## **Dataset, Data Loader, and Transforms** *(similar to HW3)*

Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Here, since our data are stored in folders by class labels, we can directly apply **torchvision.datasets.DatasetFolder** for wrapping data without much effort.

Please refer to [PyTorch official website](https://pytorch.org/vision/stable/transforms.html) for details about different transforms.

---
**The only diffference with HW3 is that the transform functions are different.**


```
### This block is similar to HW3 ###
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for food recognition.

train_tfm = transforms.Compose([
  # Resize the image into a fixed shape (height = width = 142)
	transforms.Resize((142, 142)),
  transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(15),
	transforms.RandomCrop(128),
	transforms.ToTensor(),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 142)
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

```


```
### This block is similar to HW3 ###
# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 64

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder("food-11/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder("food-11/training/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder("food-11/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
```

# **Architecture / Model Design**
The following are types of convolution layer design that has fewer parameters.

## **Depthwise & Pointwise Convolution**
![](https://i.imgur.com/FBgcA0s.png)
> Blue: the connection between layers \
> Green: the expansion of **receptive field** \
> (reference: arxiv:1810.04231)

(a) normal convolution layer: It is fully connected. The difference between fully connected layer and fully connected convolution layer is the operation. (multiply --> convolution)

(b) Depthwise convolution layer(DW): You can consider each feature map pass through their own filter and then pass through pointwise convolution layer(PW) to combine the information of all pixels in feature maps.


(c) Group convolution layer(GC): Group the feature maps. Each group passes their filter then concate together. If group_size = input_feature_size, then GC becomes DC (channels are independent). If group_size = 1, then GC becomes fully connected.

<img src="https://i.imgur.com/Hqhg0Q9.png" width="500px">


## **Implementation details**
```python
# Regular Convolution, # of params = in_chs * out_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding)

# Group Convolution, "groups" controls the connections between inputs and
# outputs. in_chs and out_chs must both be divisible by groups.
nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups)

# Depthwise Convolution, out_chs=in_chs=groups, # of params = in_chs * kernel_size^2
nn.Conv2d(in_chs, out_chs=in_chs, kernel_size, stride, padding, groups=in_chs)

# Pointwise Convolution, a.k.a 1 by 1 convolution, # of params = in_chs * out_chs
nn.Conv2d(in_chs, out_chs, 1)

# Merge Depthwise and Pointwise Convolution (without )
def dwpw_conv(in_chs, out_chs, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_chs, in_chs, kernels, stride, padding, groups=in_chs),
        nn.Conv2d(in_chs, out_chs, 1),
    )
```

## **Model**

The basic model here is simply a stack of convolutional layers followed by some fully-connected layers. You can take advatage of depthwise & pointwise convolution to make your model deeper, but still follow the size constraint.


```
class StudentNet(nn.Module):
    def __init__(self):
      super(StudentNet, self).__init__()

      # ---------- TODO ----------
      # Modify your model architecture

      self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, 3), 
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),  
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(32, 64, 3), 
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),     

        nn.Conv2d(64, 100, 3), 
        nn.BatchNorm2d(100),
        nn.ReLU(),
        nn.MaxPool2d(2, 2, 0),
        
        # Here we adopt Global Average Pooling for various input size.
        nn.AdaptiveAvgPool2d((1, 1)),
      )
      self.fc = nn.Sequential(
        nn.Linear(100, 11),
      )
      
    def forward(self, x):
      out = self.cnn(x)
      out = out.view(out.size()[0], -1)
      return self.fc(out)

```

## **Model Analysis**

Use `torchsummary` to get your model architecture (screenshot or pasting text are allowed.) and numbers of 
parameters, these two information should be submit to your NTU Cool questions.

Note that the number of parameters **should not greater than 100,000**, or you'll get penalty in this homework.



```
from torchsummary import summary

student_net = StudentNet()
summary(student_net, (3, 128, 128), device="cpu")
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1         [-1, 32, 126, 126]             896
           BatchNorm2d-2         [-1, 32, 126, 126]              64
                  ReLU-3         [-1, 32, 126, 126]               0
                Conv2d-4         [-1, 32, 124, 124]           9,248
           BatchNorm2d-5         [-1, 32, 124, 124]              64
                  ReLU-6         [-1, 32, 124, 124]               0
             MaxPool2d-7           [-1, 32, 62, 62]               0
                Conv2d-8           [-1, 64, 60, 60]          18,496
           BatchNorm2d-9           [-1, 64, 60, 60]             128
                 ReLU-10           [-1, 64, 60, 60]               0
            MaxPool2d-11           [-1, 64, 30, 30]               0
               Conv2d-12          [-1, 100, 28, 28]          57,700
          BatchNorm2d-13          [-1, 100, 28, 28]             200
                 ReLU-14          [-1, 100, 28, 28]               0
            MaxPool2d-15          [-1, 100, 14, 14]               0
    AdaptiveAvgPool2d-16            [-1, 100, 1, 1]               0
               Linear-17                   [-1, 11]           1,111
    ================================================================
    Total params: 87,907
    Trainable params: 87,907
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.19
    Forward/backward pass size (MB): 31.49
    Params size (MB): 0.34
    Estimated Total Size (MB): 32.01
    ----------------------------------------------------------------


## **Knowledge Distillation**

<img src="https://i.imgur.com/H2aF7Rv.png=100x" width="500px">

Since we have a learned big model, let it teach the other small model. In implementation, let the training target be the prediction of big model instead of the ground truth.

## **Why it works?**
* If the data is not clean, then the prediction of big model could ignore the noise of the data with wrong labeled.
* The labels might have some relations. Number 8 is more similar to 6, 9, 0 than 1, 7, for example.


## **How to implement?**
* $Loss = \alpha T^2 \times KL(\frac{\text{Teacher's Logits}}{T} || \frac{\text{Student's Logits}}{T}) + (1-\alpha)(\text{Original Loss})$
* Note that the logits here should have passed softmax.


```
def loss_fn_kd(outputs, labels, teacher_outputs, alpha=0.5):
    hard_loss = F.cross_entropy(outputs, labels) * (1. - alpha) 
    # ---------- TODO ----------
    # Complete soft loss in knowledge distillation
    soft_loss = 0 
    return hard_loss + soft_loss
```

## **Teacher Model Setting**
We provide a well-trained teacher model to help you knowledge distillation to student model.
Note that if you want to change the transform function, you should consider  if suitable for this well-trained teacher model.
* If you cannot successfully gdown, you can change a link. (Backup link is provided at the bottom of this colab tutorial).



```
# Download teacherNet
!gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output teacher_net.ckpt
# Load teacherNet
teacher_net = torch.load('./teacher_net.ckpt')
teacher_net.eval()
```

    Downloading...
    From: https://drive.google.com/uc?id=1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m
    To: /content/teacher_net.ckpt
    44.8MB [00:00, 58.6MB/s]





    ResNet(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): BasicBlock(
          (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer2): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer3): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (layer4): Sequential(
        (0): BasicBlock(
          (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (downsample): Sequential(
            (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
            (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (1): BasicBlock(
          (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (relu): ReLU(inplace=True)
          (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
      (fc): Linear(in_features=512, out_features=11, bias=True)
    )



## **Generate Pseudo Labels in Unlabeled Data**

Since we have a well-trained model, we can use this model to predict pseudo-labels and help the student network train well. Note that you 
**CANNOT** use well-trained model to pseudo-label the test data. 


---

**AGAIN, DO NOT USE TEST DATA FOR PURPOSE OTHER THAN INFERENCING**

* Because If you use teacher network to predict pseudo-labels of the test data, you can only use student network to overfit these pseudo-labels without train/unlabeled data. In this way, your kaggle accuracy will be as high as the teacher network, but the fact is that you just overfit the test data and your true testing accuracy is very low. 
* These contradict the purpose of these assignment (network compression); therefore, you should not misuse the test data.
* If you have any concerns, you can email us.



```
# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
student_net = student_net.to(device)
teacher_net = teacher_net.to(device)

# Whether to do pseudo label.
do_semi = True

def get_pseudo_labels(dataset, model):
    loader = DataLoader(dataset, batch_size=batch_size*3, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for batch in tqdm(loader):
        # A batch consists of image data and corresponding labels.
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))
            pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
        # Obtain the probability distributions by applying softmax on logits.
    pseudo_labels = torch.cat(pseudo_labels)
    # Update the labels by replacing with pseudo labels.
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset

if do_semi:
    # Generate new trainloader with unlabeled set.
    unlabeled_set = get_pseudo_labels(unlabeled_set, teacher_net)
    concat_dataset = ConcatDataset([train_set, unlabeled_set])
    train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)



```


    HBox(children=(FloatProgress(value=0.0, max=36.0), HTML(value='')))


    


## **Training** *(similar to HW3)*

You can finish supervised learning by simply running the provided code without any modification.

The function "get_pseudo_labels" is used for semi-supervised learning.
It is expected to get better performance if you use unlabeled data for semi-supervised learning.
However, you have to implement the function on your own and need to adjust several hyperparameters manually.

For more details about semi-supervised learning, please refer to [Prof. Lee's slides](https://speech.ee.ntu.edu.tw/~tlkagk/courses/ML_2016/Lecture/semi%20(v3).pdf).

Again, please notice that utilizing external data (or pre-trained model) for training is **prohibited**.

---
**The only diffference with HW3 is that you should use loss in  knowledge distillation.**





```
# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(student_net.parameters(), lr=0.0003, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 80

for epoch in range(n_epochs):
    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    student_net.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = student_net(imgs.to(device))
        # Teacher net will not be updated. And we use torch.no_grad
        # to tell torch do not retain the intermediate values
        # (which are for backpropgation) and save the memory.
        with torch.no_grad():
          soft_labels = teacher_net(imgs.to(device))
        
        # Calculate the loss in knowledge distillation method.
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")


    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    student_net.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = student_net(imgs.to(device))
          soft_labels = teacher_net(imgs.to(device))
        # We can still compute the loss (but not the gradient).
        loss = loss_fn_kd(logits, labels.to(device), soft_labels)

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs += list(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
```


    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 001/080 ] loss = 0.98192, acc = 0.32518



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 001/080 ] loss = 0.97429, acc = 0.29848



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 002/080 ] loss = 0.89496, acc = 0.39265



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 002/080 ] loss = 0.97479, acc = 0.31818



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 003/080 ] loss = 0.85041, acc = 0.42390



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 003/080 ] loss = 0.90649, acc = 0.35758



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 004/080 ] loss = 0.81575, acc = 0.45465



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 004/080 ] loss = 0.97999, acc = 0.33333



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 005/080 ] loss = 0.78934, acc = 0.47119



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 005/080 ] loss = 0.89279, acc = 0.37424



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 006/080 ] loss = 0.76561, acc = 0.48600



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 006/080 ] loss = 0.95883, acc = 0.35455



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 007/080 ] loss = 0.74605, acc = 0.49493



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 007/080 ] loss = 0.86055, acc = 0.40758



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 008/080 ] loss = 0.72489, acc = 0.51461



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 008/080 ] loss = 0.87083, acc = 0.38030



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 009/080 ] loss = 0.70889, acc = 0.52486



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 009/080 ] loss = 0.80854, acc = 0.40606



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 010/080 ] loss = 0.69723, acc = 0.53399



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 010/080 ] loss = 0.78312, acc = 0.45303



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 011/080 ] loss = 0.68075, acc = 0.54221



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 011/080 ] loss = 0.84390, acc = 0.41364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 012/080 ] loss = 0.67010, acc = 0.55235



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 012/080 ] loss = 0.96750, acc = 0.34394



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 013/080 ] loss = 0.65844, acc = 0.55783



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 013/080 ] loss = 0.82304, acc = 0.44545



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 014/080 ] loss = 0.64468, acc = 0.57183



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 014/080 ] loss = 0.75373, acc = 0.46667



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 015/080 ] loss = 0.63711, acc = 0.57884



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 015/080 ] loss = 0.77423, acc = 0.44545



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 016/080 ] loss = 0.62796, acc = 0.58320



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 016/080 ] loss = 0.77746, acc = 0.45455



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 017/080 ] loss = 0.61965, acc = 0.58746



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 017/080 ] loss = 0.85301, acc = 0.41364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 018/080 ] loss = 0.61105, acc = 0.59142



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 018/080 ] loss = 0.75065, acc = 0.50303



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 019/080 ] loss = 0.60679, acc = 0.59334



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 019/080 ] loss = 0.80715, acc = 0.45152



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 020/080 ] loss = 0.59899, acc = 0.60248



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 020/080 ] loss = 0.70264, acc = 0.51818



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 021/080 ] loss = 0.58921, acc = 0.61151



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 021/080 ] loss = 0.80058, acc = 0.47576



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 022/080 ] loss = 0.58220, acc = 0.61039



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 022/080 ] loss = 0.74697, acc = 0.48485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 023/080 ] loss = 0.57802, acc = 0.61506



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 023/080 ] loss = 0.74228, acc = 0.48333



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 024/080 ] loss = 0.56837, acc = 0.62236



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 024/080 ] loss = 0.77054, acc = 0.49848



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 025/080 ] loss = 0.56591, acc = 0.62672



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 025/080 ] loss = 0.66103, acc = 0.52879



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 026/080 ] loss = 0.55842, acc = 0.62804



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 026/080 ] loss = 0.68204, acc = 0.55000



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 027/080 ] loss = 0.55401, acc = 0.62662



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 027/080 ] loss = 0.64412, acc = 0.56212



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 028/080 ] loss = 0.54917, acc = 0.63464



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 028/080 ] loss = 0.81329, acc = 0.43485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 029/080 ] loss = 0.54511, acc = 0.63789



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 029/080 ] loss = 0.65157, acc = 0.51970



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 030/080 ] loss = 0.53784, acc = 0.64255



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 030/080 ] loss = 0.62484, acc = 0.57879



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 031/080 ] loss = 0.53563, acc = 0.64570



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 031/080 ] loss = 0.70850, acc = 0.52273



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 032/080 ] loss = 0.53304, acc = 0.64834



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 032/080 ] loss = 0.74290, acc = 0.48788



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 033/080 ] loss = 0.52511, acc = 0.65229



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 033/080 ] loss = 0.63669, acc = 0.56818



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 034/080 ] loss = 0.52440, acc = 0.65412



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 034/080 ] loss = 0.79448, acc = 0.48485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 035/080 ] loss = 0.51796, acc = 0.66051



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 035/080 ] loss = 0.66685, acc = 0.56364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 036/080 ] loss = 0.51673, acc = 0.65919



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 036/080 ] loss = 0.76977, acc = 0.51970



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 037/080 ] loss = 0.51039, acc = 0.66284



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 037/080 ] loss = 0.58990, acc = 0.59394



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 038/080 ] loss = 0.51015, acc = 0.65899



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 038/080 ] loss = 0.79827, acc = 0.48485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 039/080 ] loss = 0.50317, acc = 0.66690



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 039/080 ] loss = 0.67007, acc = 0.51364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 040/080 ] loss = 0.49964, acc = 0.66964



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 040/080 ] loss = 0.76164, acc = 0.48030



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 041/080 ] loss = 0.49698, acc = 0.66974



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 041/080 ] loss = 0.64979, acc = 0.53636



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 042/080 ] loss = 0.49099, acc = 0.67319



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 042/080 ] loss = 0.63494, acc = 0.55303



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 043/080 ] loss = 0.49168, acc = 0.67573



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 043/080 ] loss = 0.61187, acc = 0.56667



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 044/080 ] loss = 0.49331, acc = 0.67867



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 044/080 ] loss = 0.77378, acc = 0.48485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 045/080 ] loss = 0.48087, acc = 0.67918



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 045/080 ] loss = 0.61389, acc = 0.55455



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 046/080 ] loss = 0.48319, acc = 0.67705



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 046/080 ] loss = 0.63523, acc = 0.54697



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 047/080 ] loss = 0.47869, acc = 0.68009



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 047/080 ] loss = 0.66582, acc = 0.53788



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 048/080 ] loss = 0.47351, acc = 0.68547



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 048/080 ] loss = 0.77240, acc = 0.51061



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 049/080 ] loss = 0.47320, acc = 0.68862



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 049/080 ] loss = 0.70256, acc = 0.53182



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 050/080 ] loss = 0.47051, acc = 0.68720



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 050/080 ] loss = 0.59958, acc = 0.60152



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 051/080 ] loss = 0.46810, acc = 0.69267



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 051/080 ] loss = 0.67119, acc = 0.55606



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 052/080 ] loss = 0.46329, acc = 0.69754



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 052/080 ] loss = 0.57873, acc = 0.61970



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 053/080 ] loss = 0.46326, acc = 0.69663



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 053/080 ] loss = 0.78585, acc = 0.46515



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 054/080 ] loss = 0.46265, acc = 0.69257



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 054/080 ] loss = 0.62020, acc = 0.58636



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 055/080 ] loss = 0.46113, acc = 0.69572



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 055/080 ] loss = 0.57243, acc = 0.63030



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 056/080 ] loss = 0.45662, acc = 0.69988



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 056/080 ] loss = 0.69586, acc = 0.51818



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 057/080 ] loss = 0.45422, acc = 0.70008



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 057/080 ] loss = 0.61642, acc = 0.59242



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 058/080 ] loss = 0.45140, acc = 0.70140



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 058/080 ] loss = 0.68406, acc = 0.57879



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 059/080 ] loss = 0.45066, acc = 0.69947



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 059/080 ] loss = 0.72851, acc = 0.48939



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 060/080 ] loss = 0.44753, acc = 0.70495



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 060/080 ] loss = 0.65217, acc = 0.55606



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 061/080 ] loss = 0.44234, acc = 0.70789



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 061/080 ] loss = 0.61465, acc = 0.58030



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 062/080 ] loss = 0.44642, acc = 0.70231



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 062/080 ] loss = 0.71693, acc = 0.55909



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 063/080 ] loss = 0.44214, acc = 0.71094



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 063/080 ] loss = 0.58003, acc = 0.59697



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 064/080 ] loss = 0.44144, acc = 0.70678



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 064/080 ] loss = 0.54594, acc = 0.64697



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 065/080 ] loss = 0.43843, acc = 0.70749



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 065/080 ] loss = 0.60318, acc = 0.58485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 066/080 ] loss = 0.43512, acc = 0.71398



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 066/080 ] loss = 0.63196, acc = 0.54242



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 067/080 ] loss = 0.43736, acc = 0.70526



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 067/080 ] loss = 0.54518, acc = 0.58788



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 068/080 ] loss = 0.43428, acc = 0.71601



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 068/080 ] loss = 0.73762, acc = 0.53636



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 069/080 ] loss = 0.43037, acc = 0.71226



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 069/080 ] loss = 0.52712, acc = 0.63485



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 070/080 ] loss = 0.43060, acc = 0.71905



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 070/080 ] loss = 0.64272, acc = 0.59394



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 071/080 ] loss = 0.42836, acc = 0.71530



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 071/080 ] loss = 0.62219, acc = 0.60000



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 072/080 ] loss = 0.43098, acc = 0.71256



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 072/080 ] loss = 0.59485, acc = 0.61364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 073/080 ] loss = 0.42608, acc = 0.71946



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 073/080 ] loss = 0.54225, acc = 0.60606



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 074/080 ] loss = 0.42250, acc = 0.72798



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 074/080 ] loss = 0.66794, acc = 0.58333



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 075/080 ] loss = 0.42186, acc = 0.72210



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 075/080 ] loss = 0.58302, acc = 0.61364



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 076/080 ] loss = 0.42248, acc = 0.71997



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 076/080 ] loss = 0.52974, acc = 0.66061



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 077/080 ] loss = 0.41915, acc = 0.72545



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 077/080 ] loss = 0.59899, acc = 0.61818



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 078/080 ] loss = 0.41670, acc = 0.72534



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 078/080 ] loss = 0.54202, acc = 0.62273



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 079/080 ] loss = 0.41360, acc = 0.72453



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 079/080 ] loss = 0.70552, acc = 0.51667



    HBox(children=(FloatProgress(value=0.0, max=154.0), HTML(value='')))


    
    [ Train | 080/080 ] loss = 0.41217, acc = 0.72575



    HBox(children=(FloatProgress(value=0.0, max=11.0), HTML(value='')))


    
    [ Valid | 080/080 ] loss = 0.57082, acc = 0.61667


## **Testing** *(same as HW3)*

For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled ("shuffle=False" in test_loader).

Last but not least, don't forget to save the predictions into a single CSV file.
The format of CSV file should follow the rules mentioned in the slides.

### **WARNING -- Keep in Mind**

Cheating includes but not limited to:
1.   using testing labels,
2.   submitting results to previous Kaggle competitions,
3.   sharing predictions with others,
4.   copying codes from any creatures on Earth,
5.   asking other people to do it for you.

Any violations bring you punishments from getting a discount on the final grade to failing the course.

It is your responsibility to check whether your code violates the rules.
When citing codes from the Internet, you should know what these codes exactly do.
You will **NOT** be tolerated if you break the rule and claim you don't know what these codes do.



```
### This block is same as HW3 ###
# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
student_net.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = student_net(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
```


    HBox(children=(FloatProgress(value=0.0, max=53.0), HTML(value='')))


    



```
### This block is same as HW3 ###
# Save predictions into the file.
with open("predict.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")
```

## **Statistics**

|Baseline|Accuracy|Training Time|
|-|-|-|
|Simple Baseline |0.59856|2 Hours|
|Medium Baseline |0.65412|2 Hours|
|Strong Baseline |0.72819|4 Hours|
|Boss Baseline |0.81003|Unmeasueable|

## **Learning Curve**

![img](https://lh5.googleusercontent.com/amMLGa7dkqvXGmsJlrVN49VfSjClk5d-n7nCi_Y3ROK4himsBSHhB7SpdWe7Zm06ctRO77VdDkD9u_aKfAh1tMW-KcyYX7vF7LPlKqOo2fVtt3SyfsLv0KTYDB0YbAk6ZhyOIKT8Zfg)



## **Q&A**

If you have any question about this colab, please send a email to ntu-ml-2021spring-ta@googlegroups.com

## **Backup Links**


```
# resnet_model 
# !gdown --id '1zH1x39Y8a0XyOORG7TWzAnFf_YPY8e-m' --output resnet_model.ckpt
# !gdown --id '1VBIeQKH4xRHfToUxuDxtEPsqz0MHvrgd' --output resnet_model.ckpt
# !gdown --id '1Er2azErvXWS5m1jboKN7BLxNXnuAatYw' --output resnet_model.ckpt
# !gdown --id '1Qya0vmf3nRl11IyxxF7nudDpZI_Q4Amh' --output resnet_model.ckpt
# !gdown --id '1fGOOb5ndljraBIkRkLp3bW9orR4YN97U' --output resnet_model.ckpt
# !gdown --id '1apHLvZBZ3GYEMxXxToGKF7qDLn1XbOfJ' --output resnet_model.ckpt
# !gdown --id '1vsDylNsLaAqxonop7Mw3dBAig0EO7tlF' --output resnet_model.ckpt
# !gdown --id '1V_hXJM_V9-10i6wldRyl0SOiivPp4SNt' --output resnet_model.ckpt
# !gdown --id '11HzaJM2M2yg6KYhLaWpWy8WmPIIvJgnk' --output resnet_model.ckpt

# food-11
# !gdown --id '1qdyNN0Ek4S5yi-pAqHes1yjj5cNkENCc' --output food-11.zip
# !gdown --id '1c0Q1EP6yIx0O2rqVMIVInIt8wFjLxmRh' --output food-11.zip
# !gdown --id '1hKO054nT1R8egcXY2-tgQbwX4EjowRLz' --output food-11.zip
# !gdown --id '1_7_uC1WUvX6H51gQaYmI4q3AezdQJhud' --output food-11.zip
# !gdown --id '12bz82Zpx0_7BDGXq4nRt7E_fMFmILoc9' --output food-11.zip
# !gdown --id '1oiqRKrDQXVBM5y63MeEaHxFmCIzNXx1Q' --output food-11.zip
# !gdown --id '1qaL43sl4qUMeCT1OVpk4aOFycnLL5ZJX' --output food-11.zip
```
