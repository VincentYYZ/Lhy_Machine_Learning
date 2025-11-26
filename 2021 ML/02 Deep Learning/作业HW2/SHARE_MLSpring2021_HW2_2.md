# **Homework 2-2 Hessian Matrix**



## Hessian Matrix
Imagine we are training a neural network and we are trying to find out whether the model is at **local minima like, saddle point, or none of the above**. We can make our decision by calculating the Hessian matrix.

In practice, it is really hard to find a point where the gradient equals zero or all of the eigenvalues in Hessian matrix are greater than zero. In this homework, we make the following two assumptions:
1. View gradient norm less than 1e-3 as **gradient equals to zero**.
2. If minimum ratio is greater than 0.5 and gradient norm is less than 1e-3, then we assume that the model is at “local minima like”.

> Minimum ratio is defined as the proportion of positive eigenvalues.

## IMPORTANT NOTICE
In this homework, students with different student IDs will get different answers. Make sure to fill in your `student_id` in the following block correctly. Otherwise, your code may not run correctly and you will get a wrong answer.


```
student_id = 'your_student_id' # fill with your student ID

assert student_id != 'your_student_id', 'Please fill in your student_id before you start.'
```

## Calculate Hessian Matrix
The computation of Hessian is done by TA, you don't need to and shouldn't change the following code. The only thing you need to do is to run the following blocks and determine whether the model is at `local minima like`, `saddle point`, or `none of the above` according to the value of `gradient norm` and `minimum ratio`.

### Install Package to Compute Hessian.

The autograd-lib library is used to compute Hessian matrix. You can check the full document here https://github.com/cybertronai/autograd-lib.


```
!pip install autograd-lib
```

    Requirement already satisfied: autograd-lib in /usr/local/lib/python3.7/dist-packages (0.0.7)
    Requirement already satisfied: gin-config in /usr/local/lib/python3.7/dist-packages (from autograd-lib) (0.4.0)
    Requirement already satisfied: seaborn in /usr/local/lib/python3.7/dist-packages (from autograd-lib) (0.11.1)
    Requirement already satisfied: pytorch-lightning in /usr/local/lib/python3.7/dist-packages (from autograd-lib) (1.2.3)
    Requirement already satisfied: scipy>=1.0 in /usr/local/lib/python3.7/dist-packages (from seaborn->autograd-lib) (1.4.1)
    Requirement already satisfied: numpy>=1.15 in /usr/local/lib/python3.7/dist-packages (from seaborn->autograd-lib) (1.19.5)
    Requirement already satisfied: pandas>=0.23 in /usr/local/lib/python3.7/dist-packages (from seaborn->autograd-lib) (1.1.5)
    Requirement already satisfied: matplotlib>=2.2 in /usr/local/lib/python3.7/dist-packages (from seaborn->autograd-lib) (3.2.2)
    Requirement already satisfied: torch>=1.4 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (1.8.0+cu101)
    Requirement already satisfied: tensorboard>=2.2.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (2.4.1)
    Requirement already satisfied: tqdm>=4.41.0 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (4.41.1)
    Requirement already satisfied: future>=0.17.1 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (0.18.2)
    Requirement already satisfied: fsspec[http]>=0.8.1 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (0.8.7)
    Requirement already satisfied: PyYAML!=5.4.*,>=5.1 in /usr/local/lib/python3.7/dist-packages (from pytorch-lightning->autograd-lib) (5.3.1)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->seaborn->autograd-lib) (2.8.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.7/dist-packages (from pandas>=0.23->seaborn->autograd-lib) (2018.9)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn->autograd-lib) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn->autograd-lib) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from matplotlib>=2.2->seaborn->autograd-lib) (1.3.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.7/dist-packages (from torch>=1.4->pytorch-lightning->autograd-lib) (3.7.4.3)
    Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (0.4.3)
    Requirement already satisfied: wheel>=0.26; python_version >= "3" in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (0.36.2)
    Requirement already satisfied: google-auth<2,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.27.1)
    Requirement already satisfied: six>=1.10.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.15.0)
    Requirement already satisfied: protobuf>=3.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (3.12.4)
    Requirement already satisfied: grpcio>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.32.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.0.1)
    Requirement already satisfied: absl-py>=0.4 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (0.10.0)
    Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (2.23.0)
    Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (3.3.4)
    Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (54.0.0)
    Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.8.0)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /usr/local/lib/python3.7/dist-packages (from fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (3.7.0)
    Requirement already satisfied: aiohttp; extra == "http" in /usr/local/lib/python3.7/dist-packages (from fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (3.7.4.post0)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.3.0)
    Requirement already satisfied: rsa<5,>=3.1.4; python_version >= "3.6" in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (4.7.2)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (4.2.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<2,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (0.2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (1.24.3)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (2020.12.5)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata; python_version < "3.8"->fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (3.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.7/dist-packages (from aiohttp; extra == "http"->fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (5.1.0)
    Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp; extra == "http"->fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (20.3.0)
    Requirement already satisfied: async-timeout<4.0,>=3.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp; extra == "http"->fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (3.0.1)
    Requirement already satisfied: yarl<2.0,>=1.0 in /usr/local/lib/python3.7/dist-packages (from aiohttp; extra == "http"->fsspec[http]>=0.8.1->pytorch-lightning->autograd-lib) (1.6.3)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (3.1.0)
    Requirement already satisfied: pyasn1>=0.1.3 in /usr/local/lib/python3.7/dist-packages (from rsa<5,>=3.1.4; python_version >= "3.6"->google-auth<2,>=1.6.3->tensorboard>=2.2.0->pytorch-lightning->autograd-lib) (0.4.8)


### Import Libraries



```
import numpy as np
from math import pi
from collections import defaultdict
from autograd_lib import autograd_lib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")
```

### Define NN Model
The NN model here is used to fit a single variable math function.
$$f(x) = \frac{\sin(5\pi x)}{5\pi x}.$$


```
class MathRegressor(nn.Module):
    def __init__(self, num_hidden=128):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(1, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1)
        )

    def forward(self, x):
        x = self.regressor(x)
        return x
```

### Get Pretrained Checkpoints
The pretrained checkpoints is done by TA. Each student will get a different checkpoint.


```
!gdown --id 1ym6G7KKNkbsqSnMmnxdQKHO1JBoF0LPR
```

    Downloading...
    From: https://drive.google.com/uc?id=1ym6G7KKNkbsqSnMmnxdQKHO1JBoF0LPR
    To: /content/data.pth
    100% 34.5k/34.5k [00:00<00:00, 58.8MB/s]


### Load Pretrained Checkpoints and Training Data


```
# find the key from student_id
import re

key = student_id[-1]
if re.match('[0-9]', key) is not None:
    key = int(key)
else:
    key = ord(key) % 10
```


```
# load checkpoint and data corresponding to the key
model = MathRegressor()
autograd_lib.register(model)

data = torch.load('data.pth')[key]
model.load_state_dict(data['model'])
train, target = data['data']
```

### Function to compute gradient norm


```
# function to compute gradient norm
def compute_gradient_norm(model, criterion, train, target):
    model.train()
    model.zero_grad()
    output = model(train)
    loss = criterion(output, target)
    loss.backward()

    grads = []
    for p in model.regressor.children():
        if isinstance(p, nn.Linear):
            param_norm = p.weight.grad.norm(2).item()
            grads.append(param_norm)

    grad_mean = np.mean(grads) # compute mean of gradient norms

    return grad_mean
```

### Function to compute minimum ratio


```
# source code from the official document https://github.com/cybertronai/autograd-lib

# helper function to save activations
def save_activations(layer, A, _):
    '''
    A is the input of the layer, we use batch size of 6 here
    layer 1: A has size of (6, 1)
    layer 2: A has size of (6, 128)
    '''
    activations[layer] = A

# helper function to compute Hessian matrix
def compute_hess(layer, _, B):
    '''
    B is the backprop value of the layer
    layer 1: B has size of (6, 128)
    layer 2: B ahs size of (6, 1)
    '''
    A = activations[layer]
    BA = torch.einsum('nl,ni->nli', B, A) # do batch-wise outer product

    # full Hessian
    hess[layer] += torch.einsum('nli,nkj->likj', BA, BA) # do batch-wise outer product, then sum over the batch
```


```
# function to compute the minimum ratio
def compute_minimum_ratio(model, criterion, train, target):
    model.zero_grad()
    # compute Hessian matrix
    # save the gradient of each layer
    with autograd_lib.module_hook(save_activations):
        output = model(train)
        loss = criterion(output, target)

    # compute Hessian according to the gradient value stored in the previous step
    with autograd_lib.module_hook(compute_hess):
        autograd_lib.backward_hessian(output, loss='LeastSquares')

    layer_hess = list(hess.values())
    minimum_ratio = []

    # compute eigenvalues of the Hessian matrix
    for h in layer_hess:
        size = h.shape[0] * h.shape[1]
        h = h.reshape(size, size)
        h_eig = torch.symeig(h).eigenvalues # torch.symeig() returns eigenvalues and eigenvectors of a real symmetric matrix
        num_greater = torch.sum(h_eig > 0).item()
        minimum_ratio.append(num_greater / len(h_eig))

    ratio_mean = np.mean(minimum_ratio) # compute mean of minimum ratio

    return ratio_mean
```

### Mathematical Derivation

Method used here: https://en.wikipedia.org/wiki/Gauss–Newton_algorithm

> **Notations** \\
> $\mathbf{A}$: the input of the layer. \\
> $\mathbf{B}$: the backprop value. \\
> $\mathbf{Z}$: the output of the layer. \\
> $L$: the total loss, mean squared error was used here, $L=e^2$. \\
> $w$: the weight value.

Assume that the input dimension of the layer is $n$, and the output dimension of the layer is $m$.

The derivative of the loss is

\begin{align*}
    \left(\frac{\partial L}{\partial w}\right)_{nm} &= \mathbf{A}_m \mathbf{B}_n,
\end{align*}

which can be written as

\begin{align*}
    \frac{\partial L}{\partial w} &= \mathbf{B} \times \mathbf{A}.
\end{align*}

The Hessian can be derived as

\begin{align*}
    \mathbf{H}_{ij}&=\frac{\partial^2 L}{\partial w_i \partial w_j} \\
    &= \frac{\partial}{\partial w_i}\left(\frac{\partial L}{\partial w_j}\right) \\
    &= \frac{\partial}{\partial w_i}\left(\frac{2e\partial e}{\partial w_j}\right) \\
    &= 2\frac{\partial e}{\partial w_i}\frac{\partial e}{\partial w_j}+2e\frac{\partial^2 e}{\partial w_j \partial w_i}.
\end{align*}

We neglect the second-order derivative term because the term is relatively small ($e$ is small)

\begin{align*}
    \mathbf{H}_{ij}
    &\propto \frac{\partial e}{\partial w_i}\frac{\partial e}{\partial w_j},
\end{align*}

and as the error $e$ is a constant

\begin{align*}
    \mathbf{H}_{ij}
    &\propto \frac{\partial L}{\partial w_i}\frac{\partial L}{\partial w_j},
\end{align*}

then the full Hessian becomes

\begin{align*}
    \mathbf{H} &\propto (\mathbf{B}\times\mathbf{A})\times(\mathbf{B}\times\mathbf{A}).
\end{align*}



```
# the main function to compute gradient norm and minimum ratio
def main(model, train, target):
    criterion = nn.MSELoss()

    gradient_norm = compute_gradient_norm(model, criterion, train, target)
    minimum_ratio = compute_minimum_ratio(model, criterion, train, target)

    print('gradient norm: {}, minimum ratio: {}'.format(gradient_norm, minimum_ratio))
```

After running this block, you will get the value of `gradient norm` and `minimum ratio`. Determine whether the model is at `local minima like`, `saddle point`, or `none of the above`, and then submit your choice to NTU COOL.


```
if __name__ == '__main__':
    # fix random seed
    torch.manual_seed(0)

    # reset compute dictionaries
    activations = defaultdict(int)
    hess = defaultdict(float)

    # compute Hessian
    main(model, train, target)
```

    gradient norm: 0.07222428917884827, minimum ratio: 0.46484375

