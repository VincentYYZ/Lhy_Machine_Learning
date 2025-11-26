# **Homework 12 - Reinforcement Learning**

If you have any problem, e-mail us at ntu-ml-2021spring-ta@googlegroups.com



## Preliminary work

First, we need to install all necessary packages.
One of them, gym, builded by OpenAI, is a toolkit for developing Reinforcement Learning algorithm. Other packages are for visualization in colab.


```
!apt update
!apt install python-opengl xvfb -y
!pip install gym[box2d]==0.18.3 pyvirtualdisplay tqdm numpy==1.19.5 torch==1.8.1
```

    Hit:1 http://security.ubuntu.com/ubuntu bionic-security InRelease
    Ign:2 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease
    Hit:3 https://cloud.r-project.org/bin/linux/ubuntu bionic-cran40/ InRelease
    Ign:4 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  InRelease
    Hit:5 http://archive.ubuntu.com/ubuntu bionic InRelease
    Hit:6 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  Release
    Hit:7 http://ppa.launchpad.net/c2d4u.team/c2d4u4.0+/ubuntu bionic InRelease
    Hit:8 https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64  Release
    Hit:9 http://archive.ubuntu.com/ubuntu bionic-updates InRelease
    Hit:10 http://archive.ubuntu.com/ubuntu bionic-backports InRelease
    Hit:11 http://ppa.launchpad.net/cran/libgit2/ubuntu bionic InRelease
    Hit:12 http://ppa.launchpad.net/deadsnakes/ppa/ubuntu bionic InRelease
    Hit:13 http://ppa.launchpad.net/graphics-drivers/ppa/ubuntu bionic InRelease
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    86 packages can be upgraded. Run 'apt list --upgradable' to see them.
    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    python-opengl is already the newest version (3.1.0+dfsg-1).
    xvfb is already the newest version (2:1.19.6-1ubuntu4.9).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 86 not upgraded.
    Requirement already satisfied: gym[box2d] in /usr/local/lib/python3.7/dist-packages (0.17.3)
    Requirement already satisfied: pyvirtualdisplay in /usr/local/lib/python3.7/dist-packages (2.1)
    Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (4.41.1)
    Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.7/dist-packages (from gym[box2d]) (1.19.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from gym[box2d]) (1.4.1)
    Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.7/dist-packages (from gym[box2d]) (1.5.0)
    Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.7/dist-packages (from gym[box2d]) (1.3.0)
    Requirement already satisfied: box2d-py~=2.3.5; extra == "box2d" in /usr/local/lib/python3.7/dist-packages (from gym[box2d]) (2.3.8)
    Requirement already satisfied: EasyProcess in /usr/local/lib/python3.7/dist-packages (from pyvirtualdisplay) (0.3)
    Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym[box2d]) (0.16.0)



Next, set up virtual display，and import all necessaary packages.


```
%%capture
from pyvirtualdisplay import Display
virtual_display = Display(visible=0, size=(1400, 900))
virtual_display.start()

%matplotlib inline
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.notebook import tqdm
```

# Warning ! Do not revise random seed !!!
# Your submission on JudgeBoi will not reproduce your result !!!
Make your HW result to be reproducible.



```
seed = 543 # Do not change this
def fix(env, seed):
  env.seed(seed)
  env.action_space.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.set_deterministic(True)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
```

Last, call gym and build an [Lunar Lander](https://gym.openai.com/envs/LunarLander-v2/) environment.


```
%%capture
import gym
import random
env = gym.make('LunarLander-v2')
fix(env, seed) # fix the environment Do not revise this !!!
```

## What Lunar Lander？

“LunarLander-v2”is to simulate the situation when the craft lands on the surface of the moon.

This task is to enable the craft to land "safely" at the pad between the two yellow flags.
> Landing pad is always at coordinates (0,0).
> Coordinates are the first two numbers in state vector.

![](https://gym.openai.com/assets/docs/aeloop-138c89d44114492fd02822303e6b4b07213010bb14ca5856d2d49d6b62d88e53.svg)

"LunarLander-v2" actually includes "Agent" and "Environment". 

In this homework, we will utilize the function `step()` to control the action of "Agent". 

Then `step()` will return the observation/state and reward given by the "Environment".

### Observation / State

First, we can take a look at what an Observation / State looks like.


```
print(env.observation_space)
```

    Box(-inf, inf, (8,), float32)



`Box(8,)`means that observation is an 8-dim vector
### Action

Actions can be taken by looks like


```
print(env.action_space)
```

    Discrete(4)


`Discrete(4)` implies that there are four kinds of actions can be taken by agent.
- 0 implies the agent will not take any actions
- 2 implies the agent will accelerate downward
- 1, 3 implies the agent will accelerate left and right

Next, we will try to make the agent interact with the environment. 
Before taking any actions, we recommend to call `reset()` function to reset the environment. Also, this function will return the initial state of the environment.


```
initial_state = env.reset()
print(initial_state)
```

    [ 0.00396109  1.4083536   0.40119505 -0.11407257 -0.00458307 -0.09087662
      0.          0.        ]


Then, we try to get a random action from the agent's action space.


```
random_action = env.action_space.sample()
print(random_action)
```

    0


More, we can utilize `step()` to make agent act according to the randomly-selected `random_action`.
The `step()` function will return four values:
- observation / state
- reward
- done (True/ False)
- Other information


```
observation, reward, done, info = env.step(random_action)
```


```
print(done)
```

    False


### Reward


> Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. 


```
print(reward)
```

    -0.8588900517154912


### Random Agent
In the end, before we start training, we can see whether a random agent can successfully land the moon or not.


```
env.reset()

img = plt.imshow(env.render(mode='rgb_array'))

done = False
while not done:
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)

    img.set_data(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
```


    
![png](hw12_reinforcement_learning_english_version_files/hw12_reinforcement_learning_english_version_24_0.png)
    


## Policy Gradient
Now, we can build a simple policy network. The network will return one of action in the action space.


```
class PolicyGradientNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, state):
        hid = torch.tanh(self.fc1(state))
        hid = torch.tanh(self.fc2(hid))
        return F.softmax(self.fc3(hid), dim=-1)
```

Then, we need to build a simple agent. The agent will acts according to the output of the policy network above. There are a few things can be done by agent:
- `learn()`：update the policy network from log probabilities and rewards.
- `sample()`：After receiving observation from the environment, utilize policy network to tell which action to take. The return values of this function includes action and log probabilities. 


```
from torch.optim.lr_scheduler import StepLR
class PolicyGradientAgent():
    
    def __init__(self, network):
        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001)
        
    def forward(self, state):
        return self.network(state)
    def learn(self, log_probs, rewards):
        loss = (-log_probs * rewards).sum() # You don't need to revise this to pass simple baseline (but you can)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def sample(self, state):
        action_prob = self.network(torch.FloatTensor(state))
        action_dist = Categorical(action_prob)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob
```

Lastly, build a network and agent to start training.


```
network = PolicyGradientNetwork()
agent = PolicyGradientAgent(network)
```

## Trainin Agent

Now let's start to train our agent.
Through taking all the interactions between agent and environment as training data, the policy network can learn from all these attempts,


```
agent.network.train()  # Switch network into training mode 
EPISODE_PER_BATCH = 5  # update the  agent every 5 episode
NUM_BATCH = 400        # totally update the agent for 400 time

avg_total_rewards, avg_final_rewards = [], []

prg_bar = tqdm(range(NUM_BATCH))
for batch in prg_bar:

    log_probs, rewards = [], []
    total_rewards, final_rewards = [], []

    # collect trajectory
    for episode in range(EPISODE_PER_BATCH):
        
        state = env.reset()
        total_reward, total_step = 0, 0
        seq_rewards = []
        while True:

            action, log_prob = agent.sample(state) # at, log(at|st)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob) # [log(a1|s1), log(a2|s2), ...., log(at|st)]
            # seq_rewards.append(reward)
            state = next_state
            total_reward += reward
            total_step += 1
            rewards.append(reward) # change here
            # ! IMPORTANT !
            # Current reward implementation: immediate reward,  given action_list : a1, a2, a3 ......
            #                                                         rewards :     r1, r2 ,r3 ......
            # medium：change "rewards" to accumulative decaying reward, given action_list : a1,                           a2,                           a3, ......
            #                                                           rewards :           r1+0.99*r2+0.99^2*r3+......, r2+0.99*r3+0.99^2*r4+...... ,  r3+0.99*r4+0.99^2*r5+ ......
            # boss : implement DQN
            if done:
                final_rewards.append(reward)
                total_rewards.append(total_reward)
                
                break

    print(f"rewards looks like ", np.shape(rewards))  
    print(f"log_probs looks like ", np.shape(log_probs))     
    # record training process
    avg_total_reward = sum(total_rewards) / len(total_rewards)
    avg_final_reward = sum(final_rewards) / len(final_rewards)
    avg_total_rewards.append(avg_total_reward)
    avg_final_rewards.append(avg_final_reward)
    prg_bar.set_description(f"Total: {avg_total_reward: 4.1f}, Final: {avg_final_reward: 4.1f}")

    # update agent
    # rewards = np.concatenate(rewards, axis=0)
    rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-9)  # normalize the reward 
    agent.learn(torch.stack(log_probs), torch.from_numpy(rewards))
    print("logs prob looks like ", torch.stack(log_probs).size())
    print("torch.from_numpy(rewards) looks like ", torch.from_numpy(rewards).size())
```


    HBox(children=(FloatProgress(value=0.0, max=400.0), HTML(value='')))


    rewards looks like  (448,)
    log_probs looks like  (448,)
    logs prob looks like  torch.Size([448])
    torch.from_numpy(rewards) looks like  torch.Size([448])
    rewards looks like  (515,)
    log_probs looks like  (515,)
    logs prob looks like  torch.Size([515])
    torch.from_numpy(rewards) looks like  torch.Size([515])
    rewards looks like  (392,)
    log_probs looks like  (392,)
    logs prob looks like  torch.Size([392])
    torch.from_numpy(rewards) looks like  torch.Size([392])
    rewards looks like  (518,)
    log_probs looks like  (518,)
    logs prob looks like  torch.Size([518])
    torch.from_numpy(rewards) looks like  torch.Size([518])
    rewards looks like  (472,)
    log_probs looks like  (472,)
    logs prob looks like  torch.Size([472])
    torch.from_numpy(rewards) looks like  torch.Size([472])
    rewards looks like  (530,)
    log_probs looks like  (530,)
    logs prob looks like  torch.Size([530])
    torch.from_numpy(rewards) looks like  torch.Size([530])
    rewards looks like  (463,)
    log_probs looks like  (463,)
    logs prob looks like  torch.Size([463])
    torch.from_numpy(rewards) looks like  torch.Size([463])
    rewards looks like  (540,)
    log_probs looks like  (540,)
    logs prob looks like  torch.Size([540])
    torch.from_numpy(rewards) looks like  torch.Size([540])
    rewards looks like  (513,)
    log_probs looks like  (513,)
    logs prob looks like  torch.Size([513])
    torch.from_numpy(rewards) looks like  torch.Size([513])
    rewards looks like  (449,)
    log_probs looks like  (449,)
    logs prob looks like  torch.Size([449])
    torch.from_numpy(rewards) looks like  torch.Size([449])
    rewards looks like  (602,)
    log_probs looks like  (602,)
    logs prob looks like  torch.Size([602])
    torch.from_numpy(rewards) looks like  torch.Size([602])
    rewards looks like  (542,)
    log_probs looks like  (542,)
    logs prob looks like  torch.Size([542])
    torch.from_numpy(rewards) looks like  torch.Size([542])
    rewards looks like  (503,)
    log_probs looks like  (503,)
    logs prob looks like  torch.Size([503])
    torch.from_numpy(rewards) looks like  torch.Size([503])
    rewards looks like  (470,)
    log_probs looks like  (470,)
    logs prob looks like  torch.Size([470])
    torch.from_numpy(rewards) looks like  torch.Size([470])
    rewards looks like  (518,)
    log_probs looks like  (518,)
    logs prob looks like  torch.Size([518])
    torch.from_numpy(rewards) looks like  torch.Size([518])
    rewards looks like  (421,)
    log_probs looks like  (421,)
    logs prob looks like  torch.Size([421])
    torch.from_numpy(rewards) looks like  torch.Size([421])
    rewards looks like  (592,)
    log_probs looks like  (592,)
    logs prob looks like  torch.Size([592])
    torch.from_numpy(rewards) looks like  torch.Size([592])
    rewards looks like  (520,)
    log_probs looks like  (520,)
    logs prob looks like  torch.Size([520])
    torch.from_numpy(rewards) looks like  torch.Size([520])
    rewards looks like  (494,)
    log_probs looks like  (494,)
    logs prob looks like  torch.Size([494])
    torch.from_numpy(rewards) looks like  torch.Size([494])
    rewards looks like  (461,)
    log_probs looks like  (461,)
    logs prob looks like  torch.Size([461])
    torch.from_numpy(rewards) looks like  torch.Size([461])
    rewards looks like  (572,)
    log_probs looks like  (572,)
    logs prob looks like  torch.Size([572])
    torch.from_numpy(rewards) looks like  torch.Size([572])
    rewards looks like  (593,)
    log_probs looks like  (593,)
    logs prob looks like  torch.Size([593])
    torch.from_numpy(rewards) looks like  torch.Size([593])
    rewards looks like  (569,)
    log_probs looks like  (569,)
    logs prob looks like  torch.Size([569])
    torch.from_numpy(rewards) looks like  torch.Size([569])
    rewards looks like  (546,)
    log_probs looks like  (546,)
    logs prob looks like  torch.Size([546])
    torch.from_numpy(rewards) looks like  torch.Size([546])
    rewards looks like  (612,)
    log_probs looks like  (612,)
    logs prob looks like  torch.Size([612])
    torch.from_numpy(rewards) looks like  torch.Size([612])
    rewards looks like  (534,)
    log_probs looks like  (534,)
    logs prob looks like  torch.Size([534])
    torch.from_numpy(rewards) looks like  torch.Size([534])
    rewards looks like  (513,)
    log_probs looks like  (513,)
    logs prob looks like  torch.Size([513])
    torch.from_numpy(rewards) looks like  torch.Size([513])
    rewards looks like  (513,)
    log_probs looks like  (513,)
    logs prob looks like  torch.Size([513])
    torch.from_numpy(rewards) looks like  torch.Size([513])
    rewards looks like  (535,)
    log_probs looks like  (535,)
    logs prob looks like  torch.Size([535])
    torch.from_numpy(rewards) looks like  torch.Size([535])
    rewards looks like  (533,)
    log_probs looks like  (533,)
    logs prob looks like  torch.Size([533])
    torch.from_numpy(rewards) looks like  torch.Size([533])
    rewards looks like  (521,)
    log_probs looks like  (521,)
    logs prob looks like  torch.Size([521])
    torch.from_numpy(rewards) looks like  torch.Size([521])
    rewards looks like  (566,)
    log_probs looks like  (566,)
    logs prob looks like  torch.Size([566])
    torch.from_numpy(rewards) looks like  torch.Size([566])
    rewards looks like  (586,)
    log_probs looks like  (586,)
    logs prob looks like  torch.Size([586])
    torch.from_numpy(rewards) looks like  torch.Size([586])
    rewards looks like  (575,)
    log_probs looks like  (575,)
    logs prob looks like  torch.Size([575])
    torch.from_numpy(rewards) looks like  torch.Size([575])
    rewards looks like  (709,)
    log_probs looks like  (709,)
    logs prob looks like  torch.Size([709])
    torch.from_numpy(rewards) looks like  torch.Size([709])
    rewards looks like  (486,)
    log_probs looks like  (486,)
    logs prob looks like  torch.Size([486])
    torch.from_numpy(rewards) looks like  torch.Size([486])
    rewards looks like  (557,)
    log_probs looks like  (557,)
    logs prob looks like  torch.Size([557])
    torch.from_numpy(rewards) looks like  torch.Size([557])
    rewards looks like  (517,)
    log_probs looks like  (517,)
    logs prob looks like  torch.Size([517])
    torch.from_numpy(rewards) looks like  torch.Size([517])
    rewards looks like  (550,)
    log_probs looks like  (550,)
    logs prob looks like  torch.Size([550])
    torch.from_numpy(rewards) looks like  torch.Size([550])
    rewards looks like  (690,)
    log_probs looks like  (690,)
    logs prob looks like  torch.Size([690])
    torch.from_numpy(rewards) looks like  torch.Size([690])
    rewards looks like  (591,)
    log_probs looks like  (591,)
    logs prob looks like  torch.Size([591])
    torch.from_numpy(rewards) looks like  torch.Size([591])
    rewards looks like  (689,)
    log_probs looks like  (689,)
    logs prob looks like  torch.Size([689])
    torch.from_numpy(rewards) looks like  torch.Size([689])
    rewards looks like  (1059,)
    log_probs looks like  (1059,)
    logs prob looks like  torch.Size([1059])
    torch.from_numpy(rewards) looks like  torch.Size([1059])
    rewards looks like  (619,)
    log_probs looks like  (619,)
    logs prob looks like  torch.Size([619])
    torch.from_numpy(rewards) looks like  torch.Size([619])
    rewards looks like  (527,)
    log_probs looks like  (527,)
    logs prob looks like  torch.Size([527])
    torch.from_numpy(rewards) looks like  torch.Size([527])
    rewards looks like  (514,)
    log_probs looks like  (514,)
    logs prob looks like  torch.Size([514])
    torch.from_numpy(rewards) looks like  torch.Size([514])
    rewards looks like  (655,)
    log_probs looks like  (655,)
    logs prob looks like  torch.Size([655])
    torch.from_numpy(rewards) looks like  torch.Size([655])
    rewards looks like  (667,)
    log_probs looks like  (667,)
    logs prob looks like  torch.Size([667])
    torch.from_numpy(rewards) looks like  torch.Size([667])
    rewards looks like  (712,)
    log_probs looks like  (712,)
    logs prob looks like  torch.Size([712])
    torch.from_numpy(rewards) looks like  torch.Size([712])
    rewards looks like  (636,)
    log_probs looks like  (636,)
    logs prob looks like  torch.Size([636])
    torch.from_numpy(rewards) looks like  torch.Size([636])
    rewards looks like  (620,)
    log_probs looks like  (620,)
    logs prob looks like  torch.Size([620])
    torch.from_numpy(rewards) looks like  torch.Size([620])
    rewards looks like  (543,)
    log_probs looks like  (543,)
    logs prob looks like  torch.Size([543])
    torch.from_numpy(rewards) looks like  torch.Size([543])
    rewards looks like  (586,)
    log_probs looks like  (586,)
    logs prob looks like  torch.Size([586])
    torch.from_numpy(rewards) looks like  torch.Size([586])
    rewards looks like  (498,)
    log_probs looks like  (498,)
    logs prob looks like  torch.Size([498])
    torch.from_numpy(rewards) looks like  torch.Size([498])
    rewards looks like  (586,)
    log_probs looks like  (586,)
    logs prob looks like  torch.Size([586])
    torch.from_numpy(rewards) looks like  torch.Size([586])
    rewards looks like  (591,)
    log_probs looks like  (591,)
    logs prob looks like  torch.Size([591])
    torch.from_numpy(rewards) looks like  torch.Size([591])
    rewards looks like  (693,)
    log_probs looks like  (693,)
    logs prob looks like  torch.Size([693])
    torch.from_numpy(rewards) looks like  torch.Size([693])
    rewards looks like  (648,)
    log_probs looks like  (648,)
    logs prob looks like  torch.Size([648])
    torch.from_numpy(rewards) looks like  torch.Size([648])
    rewards looks like  (513,)
    log_probs looks like  (513,)
    logs prob looks like  torch.Size([513])
    torch.from_numpy(rewards) looks like  torch.Size([513])
    rewards looks like  (574,)
    log_probs looks like  (574,)
    logs prob looks like  torch.Size([574])
    torch.from_numpy(rewards) looks like  torch.Size([574])
    rewards looks like  (718,)
    log_probs looks like  (718,)
    logs prob looks like  torch.Size([718])
    torch.from_numpy(rewards) looks like  torch.Size([718])
    rewards looks like  (730,)
    log_probs looks like  (730,)
    logs prob looks like  torch.Size([730])
    torch.from_numpy(rewards) looks like  torch.Size([730])
    rewards looks like  (668,)
    log_probs looks like  (668,)
    logs prob looks like  torch.Size([668])
    torch.from_numpy(rewards) looks like  torch.Size([668])
    rewards looks like  (754,)
    log_probs looks like  (754,)
    logs prob looks like  torch.Size([754])
    torch.from_numpy(rewards) looks like  torch.Size([754])
    rewards looks like  (712,)
    log_probs looks like  (712,)
    logs prob looks like  torch.Size([712])
    torch.from_numpy(rewards) looks like  torch.Size([712])
    rewards looks like  (470,)
    log_probs looks like  (470,)
    logs prob looks like  torch.Size([470])
    torch.from_numpy(rewards) looks like  torch.Size([470])
    rewards looks like  (665,)
    log_probs looks like  (665,)
    logs prob looks like  torch.Size([665])
    torch.from_numpy(rewards) looks like  torch.Size([665])
    rewards looks like  (585,)
    log_probs looks like  (585,)
    logs prob looks like  torch.Size([585])
    torch.from_numpy(rewards) looks like  torch.Size([585])
    rewards looks like  (512,)
    log_probs looks like  (512,)
    logs prob looks like  torch.Size([512])
    torch.from_numpy(rewards) looks like  torch.Size([512])
    rewards looks like  (702,)
    log_probs looks like  (702,)
    logs prob looks like  torch.Size([702])
    torch.from_numpy(rewards) looks like  torch.Size([702])
    rewards looks like  (596,)
    log_probs looks like  (596,)
    logs prob looks like  torch.Size([596])
    torch.from_numpy(rewards) looks like  torch.Size([596])
    rewards looks like  (626,)
    log_probs looks like  (626,)
    logs prob looks like  torch.Size([626])
    torch.from_numpy(rewards) looks like  torch.Size([626])
    rewards looks like  (566,)
    log_probs looks like  (566,)
    logs prob looks like  torch.Size([566])
    torch.from_numpy(rewards) looks like  torch.Size([566])
    rewards looks like  (717,)
    log_probs looks like  (717,)
    logs prob looks like  torch.Size([717])
    torch.from_numpy(rewards) looks like  torch.Size([717])
    rewards looks like  (708,)
    log_probs looks like  (708,)
    logs prob looks like  torch.Size([708])
    torch.from_numpy(rewards) looks like  torch.Size([708])
    rewards looks like  (565,)
    log_probs looks like  (565,)
    logs prob looks like  torch.Size([565])
    torch.from_numpy(rewards) looks like  torch.Size([565])
    rewards looks like  (450,)
    log_probs looks like  (450,)
    logs prob looks like  torch.Size([450])
    torch.from_numpy(rewards) looks like  torch.Size([450])
    rewards looks like  (584,)
    log_probs looks like  (584,)
    logs prob looks like  torch.Size([584])
    torch.from_numpy(rewards) looks like  torch.Size([584])
    rewards looks like  (670,)
    log_probs looks like  (670,)
    logs prob looks like  torch.Size([670])
    torch.from_numpy(rewards) looks like  torch.Size([670])
    rewards looks like  (691,)
    log_probs looks like  (691,)
    logs prob looks like  torch.Size([691])
    torch.from_numpy(rewards) looks like  torch.Size([691])
    rewards looks like  (760,)
    log_probs looks like  (760,)
    logs prob looks like  torch.Size([760])
    torch.from_numpy(rewards) looks like  torch.Size([760])
    rewards looks like  (752,)
    log_probs looks like  (752,)
    logs prob looks like  torch.Size([752])
    torch.from_numpy(rewards) looks like  torch.Size([752])
    rewards looks like  (478,)
    log_probs looks like  (478,)
    logs prob looks like  torch.Size([478])
    torch.from_numpy(rewards) looks like  torch.Size([478])
    rewards looks like  (553,)
    log_probs looks like  (553,)
    logs prob looks like  torch.Size([553])
    torch.from_numpy(rewards) looks like  torch.Size([553])
    rewards looks like  (1660,)
    log_probs looks like  (1660,)
    logs prob looks like  torch.Size([1660])
    torch.from_numpy(rewards) looks like  torch.Size([1660])
    rewards looks like  (751,)
    log_probs looks like  (751,)
    logs prob looks like  torch.Size([751])
    torch.from_numpy(rewards) looks like  torch.Size([751])
    rewards looks like  (801,)
    log_probs looks like  (801,)
    logs prob looks like  torch.Size([801])
    torch.from_numpy(rewards) looks like  torch.Size([801])
    rewards looks like  (715,)
    log_probs looks like  (715,)
    logs prob looks like  torch.Size([715])
    torch.from_numpy(rewards) looks like  torch.Size([715])
    rewards looks like  (708,)
    log_probs looks like  (708,)
    logs prob looks like  torch.Size([708])
    torch.from_numpy(rewards) looks like  torch.Size([708])
    rewards looks like  (609,)
    log_probs looks like  (609,)
    logs prob looks like  torch.Size([609])
    torch.from_numpy(rewards) looks like  torch.Size([609])
    rewards looks like  (732,)
    log_probs looks like  (732,)
    logs prob looks like  torch.Size([732])
    torch.from_numpy(rewards) looks like  torch.Size([732])
    rewards looks like  (603,)
    log_probs looks like  (603,)
    logs prob looks like  torch.Size([603])
    torch.from_numpy(rewards) looks like  torch.Size([603])
    rewards looks like  (603,)
    log_probs looks like  (603,)
    logs prob looks like  torch.Size([603])
    torch.from_numpy(rewards) looks like  torch.Size([603])
    rewards looks like  (665,)
    log_probs looks like  (665,)
    logs prob looks like  torch.Size([665])
    torch.from_numpy(rewards) looks like  torch.Size([665])
    rewards looks like  (658,)
    log_probs looks like  (658,)
    logs prob looks like  torch.Size([658])
    torch.from_numpy(rewards) looks like  torch.Size([658])
    rewards looks like  (783,)
    log_probs looks like  (783,)
    logs prob looks like  torch.Size([783])
    torch.from_numpy(rewards) looks like  torch.Size([783])
    rewards looks like  (652,)
    log_probs looks like  (652,)
    logs prob looks like  torch.Size([652])
    torch.from_numpy(rewards) looks like  torch.Size([652])
    rewards looks like  (892,)
    log_probs looks like  (892,)
    logs prob looks like  torch.Size([892])
    torch.from_numpy(rewards) looks like  torch.Size([892])
    rewards looks like  (821,)
    log_probs looks like  (821,)
    logs prob looks like  torch.Size([821])
    torch.from_numpy(rewards) looks like  torch.Size([821])
    rewards looks like  (986,)
    log_probs looks like  (986,)
    logs prob looks like  torch.Size([986])
    torch.from_numpy(rewards) looks like  torch.Size([986])
    rewards looks like  (916,)
    log_probs looks like  (916,)
    logs prob looks like  torch.Size([916])
    torch.from_numpy(rewards) looks like  torch.Size([916])
    rewards looks like  (742,)
    log_probs looks like  (742,)
    logs prob looks like  torch.Size([742])
    torch.from_numpy(rewards) looks like  torch.Size([742])
    rewards looks like  (604,)
    log_probs looks like  (604,)
    logs prob looks like  torch.Size([604])
    torch.from_numpy(rewards) looks like  torch.Size([604])
    rewards looks like  (818,)
    log_probs looks like  (818,)
    logs prob looks like  torch.Size([818])
    torch.from_numpy(rewards) looks like  torch.Size([818])
    rewards looks like  (855,)
    log_probs looks like  (855,)
    logs prob looks like  torch.Size([855])
    torch.from_numpy(rewards) looks like  torch.Size([855])
    rewards looks like  (795,)
    log_probs looks like  (795,)
    logs prob looks like  torch.Size([795])
    torch.from_numpy(rewards) looks like  torch.Size([795])
    rewards looks like  (868,)
    log_probs looks like  (868,)
    logs prob looks like  torch.Size([868])
    torch.from_numpy(rewards) looks like  torch.Size([868])
    rewards looks like  (800,)
    log_probs looks like  (800,)
    logs prob looks like  torch.Size([800])
    torch.from_numpy(rewards) looks like  torch.Size([800])
    rewards looks like  (820,)
    log_probs looks like  (820,)
    logs prob looks like  torch.Size([820])
    torch.from_numpy(rewards) looks like  torch.Size([820])
    rewards looks like  (760,)
    log_probs looks like  (760,)
    logs prob looks like  torch.Size([760])
    torch.from_numpy(rewards) looks like  torch.Size([760])
    rewards looks like  (886,)
    log_probs looks like  (886,)
    logs prob looks like  torch.Size([886])
    torch.from_numpy(rewards) looks like  torch.Size([886])
    rewards looks like  (1027,)
    log_probs looks like  (1027,)
    logs prob looks like  torch.Size([1027])
    torch.from_numpy(rewards) looks like  torch.Size([1027])
    rewards looks like  (819,)
    log_probs looks like  (819,)
    logs prob looks like  torch.Size([819])
    torch.from_numpy(rewards) looks like  torch.Size([819])
    rewards looks like  (934,)
    log_probs looks like  (934,)
    logs prob looks like  torch.Size([934])
    torch.from_numpy(rewards) looks like  torch.Size([934])
    rewards looks like  (1648,)
    log_probs looks like  (1648,)
    logs prob looks like  torch.Size([1648])
    torch.from_numpy(rewards) looks like  torch.Size([1648])
    rewards looks like  (1057,)
    log_probs looks like  (1057,)
    logs prob looks like  torch.Size([1057])
    torch.from_numpy(rewards) looks like  torch.Size([1057])
    rewards looks like  (861,)
    log_probs looks like  (861,)
    logs prob looks like  torch.Size([861])
    torch.from_numpy(rewards) looks like  torch.Size([861])
    rewards looks like  (1533,)
    log_probs looks like  (1533,)
    logs prob looks like  torch.Size([1533])
    torch.from_numpy(rewards) looks like  torch.Size([1533])
    rewards looks like  (920,)
    log_probs looks like  (920,)
    logs prob looks like  torch.Size([920])
    torch.from_numpy(rewards) looks like  torch.Size([920])
    rewards looks like  (905,)
    log_probs looks like  (905,)
    logs prob looks like  torch.Size([905])
    torch.from_numpy(rewards) looks like  torch.Size([905])
    rewards looks like  (814,)
    log_probs looks like  (814,)
    logs prob looks like  torch.Size([814])
    torch.from_numpy(rewards) looks like  torch.Size([814])
    rewards looks like  (809,)
    log_probs looks like  (809,)
    logs prob looks like  torch.Size([809])
    torch.from_numpy(rewards) looks like  torch.Size([809])
    rewards looks like  (873,)
    log_probs looks like  (873,)
    logs prob looks like  torch.Size([873])
    torch.from_numpy(rewards) looks like  torch.Size([873])
    rewards looks like  (727,)
    log_probs looks like  (727,)
    logs prob looks like  torch.Size([727])
    torch.from_numpy(rewards) looks like  torch.Size([727])
    rewards looks like  (1129,)
    log_probs looks like  (1129,)
    logs prob looks like  torch.Size([1129])
    torch.from_numpy(rewards) looks like  torch.Size([1129])
    rewards looks like  (1394,)
    log_probs looks like  (1394,)
    logs prob looks like  torch.Size([1394])
    torch.from_numpy(rewards) looks like  torch.Size([1394])
    rewards looks like  (884,)
    log_probs looks like  (884,)
    logs prob looks like  torch.Size([884])
    torch.from_numpy(rewards) looks like  torch.Size([884])
    rewards looks like  (1132,)
    log_probs looks like  (1132,)
    logs prob looks like  torch.Size([1132])
    torch.from_numpy(rewards) looks like  torch.Size([1132])
    rewards looks like  (1007,)
    log_probs looks like  (1007,)
    logs prob looks like  torch.Size([1007])
    torch.from_numpy(rewards) looks like  torch.Size([1007])
    rewards looks like  (711,)
    log_probs looks like  (711,)
    logs prob looks like  torch.Size([711])
    torch.from_numpy(rewards) looks like  torch.Size([711])
    rewards looks like  (836,)
    log_probs looks like  (836,)
    logs prob looks like  torch.Size([836])
    torch.from_numpy(rewards) looks like  torch.Size([836])
    rewards looks like  (1514,)
    log_probs looks like  (1514,)
    logs prob looks like  torch.Size([1514])
    torch.from_numpy(rewards) looks like  torch.Size([1514])
    rewards looks like  (896,)
    log_probs looks like  (896,)
    logs prob looks like  torch.Size([896])
    torch.from_numpy(rewards) looks like  torch.Size([896])
    rewards looks like  (912,)
    log_probs looks like  (912,)
    logs prob looks like  torch.Size([912])
    torch.from_numpy(rewards) looks like  torch.Size([912])
    rewards looks like  (1478,)
    log_probs looks like  (1478,)
    logs prob looks like  torch.Size([1478])
    torch.from_numpy(rewards) looks like  torch.Size([1478])
    rewards looks like  (1279,)
    log_probs looks like  (1279,)
    logs prob looks like  torch.Size([1279])
    torch.from_numpy(rewards) looks like  torch.Size([1279])
    rewards looks like  (676,)
    log_probs looks like  (676,)
    logs prob looks like  torch.Size([676])
    torch.from_numpy(rewards) looks like  torch.Size([676])
    rewards looks like  (1768,)
    log_probs looks like  (1768,)
    logs prob looks like  torch.Size([1768])
    torch.from_numpy(rewards) looks like  torch.Size([1768])
    rewards looks like  (897,)
    log_probs looks like  (897,)
    logs prob looks like  torch.Size([897])
    torch.from_numpy(rewards) looks like  torch.Size([897])
    rewards looks like  (1119,)
    log_probs looks like  (1119,)
    logs prob looks like  torch.Size([1119])
    torch.from_numpy(rewards) looks like  torch.Size([1119])
    rewards looks like  (943,)
    log_probs looks like  (943,)
    logs prob looks like  torch.Size([943])
    torch.from_numpy(rewards) looks like  torch.Size([943])
    rewards looks like  (1255,)
    log_probs looks like  (1255,)
    logs prob looks like  torch.Size([1255])
    torch.from_numpy(rewards) looks like  torch.Size([1255])
    rewards looks like  (861,)
    log_probs looks like  (861,)
    logs prob looks like  torch.Size([861])
    torch.from_numpy(rewards) looks like  torch.Size([861])
    rewards looks like  (1149,)
    log_probs looks like  (1149,)
    logs prob looks like  torch.Size([1149])
    torch.from_numpy(rewards) looks like  torch.Size([1149])
    rewards looks like  (1229,)
    log_probs looks like  (1229,)
    logs prob looks like  torch.Size([1229])
    torch.from_numpy(rewards) looks like  torch.Size([1229])
    rewards looks like  (1680,)
    log_probs looks like  (1680,)
    logs prob looks like  torch.Size([1680])
    torch.from_numpy(rewards) looks like  torch.Size([1680])
    rewards looks like  (1731,)
    log_probs looks like  (1731,)
    logs prob looks like  torch.Size([1731])
    torch.from_numpy(rewards) looks like  torch.Size([1731])
    rewards looks like  (1017,)
    log_probs looks like  (1017,)
    logs prob looks like  torch.Size([1017])
    torch.from_numpy(rewards) looks like  torch.Size([1017])
    rewards looks like  (990,)
    log_probs looks like  (990,)
    logs prob looks like  torch.Size([990])
    torch.from_numpy(rewards) looks like  torch.Size([990])
    rewards looks like  (1020,)
    log_probs looks like  (1020,)
    logs prob looks like  torch.Size([1020])
    torch.from_numpy(rewards) looks like  torch.Size([1020])
    rewards looks like  (1240,)
    log_probs looks like  (1240,)
    logs prob looks like  torch.Size([1240])
    torch.from_numpy(rewards) looks like  torch.Size([1240])
    rewards looks like  (774,)
    log_probs looks like  (774,)
    logs prob looks like  torch.Size([774])
    torch.from_numpy(rewards) looks like  torch.Size([774])
    rewards looks like  (1069,)
    log_probs looks like  (1069,)
    logs prob looks like  torch.Size([1069])
    torch.from_numpy(rewards) looks like  torch.Size([1069])
    rewards looks like  (1355,)
    log_probs looks like  (1355,)
    logs prob looks like  torch.Size([1355])
    torch.from_numpy(rewards) looks like  torch.Size([1355])
    rewards looks like  (1556,)
    log_probs looks like  (1556,)
    logs prob looks like  torch.Size([1556])
    torch.from_numpy(rewards) looks like  torch.Size([1556])
    rewards looks like  (1840,)
    log_probs looks like  (1840,)
    logs prob looks like  torch.Size([1840])
    torch.from_numpy(rewards) looks like  torch.Size([1840])
    rewards looks like  (1352,)
    log_probs looks like  (1352,)
    logs prob looks like  torch.Size([1352])
    torch.from_numpy(rewards) looks like  torch.Size([1352])
    rewards looks like  (1617,)
    log_probs looks like  (1617,)
    logs prob looks like  torch.Size([1617])
    torch.from_numpy(rewards) looks like  torch.Size([1617])
    rewards looks like  (1637,)
    log_probs looks like  (1637,)
    logs prob looks like  torch.Size([1637])
    torch.from_numpy(rewards) looks like  torch.Size([1637])
    rewards looks like  (1606,)
    log_probs looks like  (1606,)
    logs prob looks like  torch.Size([1606])
    torch.from_numpy(rewards) looks like  torch.Size([1606])
    rewards looks like  (860,)
    log_probs looks like  (860,)
    logs prob looks like  torch.Size([860])
    torch.from_numpy(rewards) looks like  torch.Size([860])
    rewards looks like  (1780,)
    log_probs looks like  (1780,)
    logs prob looks like  torch.Size([1780])
    torch.from_numpy(rewards) looks like  torch.Size([1780])
    rewards looks like  (2248,)
    log_probs looks like  (2248,)
    logs prob looks like  torch.Size([2248])
    torch.from_numpy(rewards) looks like  torch.Size([2248])
    rewards looks like  (1410,)
    log_probs looks like  (1410,)
    logs prob looks like  torch.Size([1410])
    torch.from_numpy(rewards) looks like  torch.Size([1410])
    rewards looks like  (557,)
    log_probs looks like  (557,)
    logs prob looks like  torch.Size([557])
    torch.from_numpy(rewards) looks like  torch.Size([557])
    rewards looks like  (719,)
    log_probs looks like  (719,)
    logs prob looks like  torch.Size([719])
    torch.from_numpy(rewards) looks like  torch.Size([719])
    rewards looks like  (1919,)
    log_probs looks like  (1919,)
    logs prob looks like  torch.Size([1919])
    torch.from_numpy(rewards) looks like  torch.Size([1919])
    rewards looks like  (1250,)
    log_probs looks like  (1250,)
    logs prob looks like  torch.Size([1250])
    torch.from_numpy(rewards) looks like  torch.Size([1250])
    rewards looks like  (1054,)
    log_probs looks like  (1054,)
    logs prob looks like  torch.Size([1054])
    torch.from_numpy(rewards) looks like  torch.Size([1054])
    rewards looks like  (1276,)
    log_probs looks like  (1276,)
    logs prob looks like  torch.Size([1276])
    torch.from_numpy(rewards) looks like  torch.Size([1276])
    rewards looks like  (1040,)
    log_probs looks like  (1040,)
    logs prob looks like  torch.Size([1040])
    torch.from_numpy(rewards) looks like  torch.Size([1040])
    rewards looks like  (991,)
    log_probs looks like  (991,)
    logs prob looks like  torch.Size([991])
    torch.from_numpy(rewards) looks like  torch.Size([991])
    rewards looks like  (1390,)
    log_probs looks like  (1390,)
    logs prob looks like  torch.Size([1390])
    torch.from_numpy(rewards) looks like  torch.Size([1390])
    rewards looks like  (1349,)
    log_probs looks like  (1349,)
    logs prob looks like  torch.Size([1349])
    torch.from_numpy(rewards) looks like  torch.Size([1349])
    rewards looks like  (1332,)
    log_probs looks like  (1332,)
    logs prob looks like  torch.Size([1332])
    torch.from_numpy(rewards) looks like  torch.Size([1332])
    rewards looks like  (1378,)
    log_probs looks like  (1378,)
    logs prob looks like  torch.Size([1378])
    torch.from_numpy(rewards) looks like  torch.Size([1378])
    rewards looks like  (1967,)
    log_probs looks like  (1967,)
    logs prob looks like  torch.Size([1967])
    torch.from_numpy(rewards) looks like  torch.Size([1967])
    rewards looks like  (1789,)
    log_probs looks like  (1789,)
    logs prob looks like  torch.Size([1789])
    torch.from_numpy(rewards) looks like  torch.Size([1789])
    rewards looks like  (1325,)
    log_probs looks like  (1325,)
    logs prob looks like  torch.Size([1325])
    torch.from_numpy(rewards) looks like  torch.Size([1325])
    rewards looks like  (1685,)
    log_probs looks like  (1685,)
    logs prob looks like  torch.Size([1685])
    torch.from_numpy(rewards) looks like  torch.Size([1685])
    rewards looks like  (1895,)
    log_probs looks like  (1895,)
    logs prob looks like  torch.Size([1895])
    torch.from_numpy(rewards) looks like  torch.Size([1895])
    rewards looks like  (1920,)
    log_probs looks like  (1920,)
    logs prob looks like  torch.Size([1920])
    torch.from_numpy(rewards) looks like  torch.Size([1920])
    rewards looks like  (1522,)
    log_probs looks like  (1522,)
    logs prob looks like  torch.Size([1522])
    torch.from_numpy(rewards) looks like  torch.Size([1522])
    rewards looks like  (1173,)
    log_probs looks like  (1173,)
    logs prob looks like  torch.Size([1173])
    torch.from_numpy(rewards) looks like  torch.Size([1173])
    rewards looks like  (2136,)
    log_probs looks like  (2136,)
    logs prob looks like  torch.Size([2136])
    torch.from_numpy(rewards) looks like  torch.Size([2136])
    rewards looks like  (1696,)
    log_probs looks like  (1696,)
    logs prob looks like  torch.Size([1696])
    torch.from_numpy(rewards) looks like  torch.Size([1696])
    rewards looks like  (568,)
    log_probs looks like  (568,)
    logs prob looks like  torch.Size([568])
    torch.from_numpy(rewards) looks like  torch.Size([568])
    rewards looks like  (1475,)
    log_probs looks like  (1475,)
    logs prob looks like  torch.Size([1475])
    torch.from_numpy(rewards) looks like  torch.Size([1475])
    rewards looks like  (2470,)
    log_probs looks like  (2470,)
    logs prob looks like  torch.Size([2470])
    torch.from_numpy(rewards) looks like  torch.Size([2470])
    rewards looks like  (3053,)
    log_probs looks like  (3053,)
    logs prob looks like  torch.Size([3053])
    torch.from_numpy(rewards) looks like  torch.Size([3053])
    rewards looks like  (915,)
    log_probs looks like  (915,)
    logs prob looks like  torch.Size([915])
    torch.from_numpy(rewards) looks like  torch.Size([915])
    rewards looks like  (2049,)
    log_probs looks like  (2049,)
    logs prob looks like  torch.Size([2049])
    torch.from_numpy(rewards) looks like  torch.Size([2049])
    rewards looks like  (2068,)
    log_probs looks like  (2068,)
    logs prob looks like  torch.Size([2068])
    torch.from_numpy(rewards) looks like  torch.Size([2068])
    rewards looks like  (2528,)
    log_probs looks like  (2528,)
    logs prob looks like  torch.Size([2528])
    torch.from_numpy(rewards) looks like  torch.Size([2528])
    rewards looks like  (1839,)
    log_probs looks like  (1839,)
    logs prob looks like  torch.Size([1839])
    torch.from_numpy(rewards) looks like  torch.Size([1839])
    rewards looks like  (497,)
    log_probs looks like  (497,)
    logs prob looks like  torch.Size([497])
    torch.from_numpy(rewards) looks like  torch.Size([497])
    rewards looks like  (627,)
    log_probs looks like  (627,)
    logs prob looks like  torch.Size([627])
    torch.from_numpy(rewards) looks like  torch.Size([627])
    rewards looks like  (2354,)
    log_probs looks like  (2354,)
    logs prob looks like  torch.Size([2354])
    torch.from_numpy(rewards) looks like  torch.Size([2354])
    rewards looks like  (2394,)
    log_probs looks like  (2394,)
    logs prob looks like  torch.Size([2394])
    torch.from_numpy(rewards) looks like  torch.Size([2394])
    rewards looks like  (743,)
    log_probs looks like  (743,)
    logs prob looks like  torch.Size([743])
    torch.from_numpy(rewards) looks like  torch.Size([743])
    rewards looks like  (1572,)
    log_probs looks like  (1572,)
    logs prob looks like  torch.Size([1572])
    torch.from_numpy(rewards) looks like  torch.Size([1572])
    rewards looks like  (2575,)
    log_probs looks like  (2575,)
    logs prob looks like  torch.Size([2575])
    torch.from_numpy(rewards) looks like  torch.Size([2575])
    rewards looks like  (2226,)
    log_probs looks like  (2226,)
    logs prob looks like  torch.Size([2226])
    torch.from_numpy(rewards) looks like  torch.Size([2226])
    rewards looks like  (541,)
    log_probs looks like  (541,)
    logs prob looks like  torch.Size([541])
    torch.from_numpy(rewards) looks like  torch.Size([541])
    rewards looks like  (820,)
    log_probs looks like  (820,)
    logs prob looks like  torch.Size([820])
    torch.from_numpy(rewards) looks like  torch.Size([820])
    rewards looks like  (2584,)
    log_probs looks like  (2584,)
    logs prob looks like  torch.Size([2584])
    torch.from_numpy(rewards) looks like  torch.Size([2584])
    rewards looks like  (1792,)
    log_probs looks like  (1792,)
    logs prob looks like  torch.Size([1792])
    torch.from_numpy(rewards) looks like  torch.Size([1792])
    rewards looks like  (1613,)
    log_probs looks like  (1613,)
    logs prob looks like  torch.Size([1613])
    torch.from_numpy(rewards) looks like  torch.Size([1613])
    rewards looks like  (4300,)
    log_probs looks like  (4300,)
    logs prob looks like  torch.Size([4300])
    torch.from_numpy(rewards) looks like  torch.Size([4300])
    rewards looks like  (1602,)
    log_probs looks like  (1602,)
    logs prob looks like  torch.Size([1602])
    torch.from_numpy(rewards) looks like  torch.Size([1602])
    rewards looks like  (3313,)
    log_probs looks like  (3313,)
    logs prob looks like  torch.Size([3313])
    torch.from_numpy(rewards) looks like  torch.Size([3313])
    rewards looks like  (1538,)
    log_probs looks like  (1538,)
    logs prob looks like  torch.Size([1538])
    torch.from_numpy(rewards) looks like  torch.Size([1538])
    rewards looks like  (1824,)
    log_probs looks like  (1824,)
    logs prob looks like  torch.Size([1824])
    torch.from_numpy(rewards) looks like  torch.Size([1824])
    rewards looks like  (1320,)
    log_probs looks like  (1320,)
    logs prob looks like  torch.Size([1320])
    torch.from_numpy(rewards) looks like  torch.Size([1320])
    rewards looks like  (2077,)
    log_probs looks like  (2077,)
    logs prob looks like  torch.Size([2077])
    torch.from_numpy(rewards) looks like  torch.Size([2077])
    rewards looks like  (1995,)
    log_probs looks like  (1995,)
    logs prob looks like  torch.Size([1995])
    torch.from_numpy(rewards) looks like  torch.Size([1995])
    rewards looks like  (1089,)
    log_probs looks like  (1089,)
    logs prob looks like  torch.Size([1089])
    torch.from_numpy(rewards) looks like  torch.Size([1089])
    rewards looks like  (1135,)
    log_probs looks like  (1135,)
    logs prob looks like  torch.Size([1135])
    torch.from_numpy(rewards) looks like  torch.Size([1135])
    rewards looks like  (1617,)
    log_probs looks like  (1617,)
    logs prob looks like  torch.Size([1617])
    torch.from_numpy(rewards) looks like  torch.Size([1617])
    rewards looks like  (942,)
    log_probs looks like  (942,)
    logs prob looks like  torch.Size([942])
    torch.from_numpy(rewards) looks like  torch.Size([942])
    rewards looks like  (2006,)
    log_probs looks like  (2006,)
    logs prob looks like  torch.Size([2006])
    torch.from_numpy(rewards) looks like  torch.Size([2006])
    rewards looks like  (2204,)
    log_probs looks like  (2204,)
    logs prob looks like  torch.Size([2204])
    torch.from_numpy(rewards) looks like  torch.Size([2204])
    rewards looks like  (1060,)
    log_probs looks like  (1060,)
    logs prob looks like  torch.Size([1060])
    torch.from_numpy(rewards) looks like  torch.Size([1060])
    rewards looks like  (1994,)
    log_probs looks like  (1994,)
    logs prob looks like  torch.Size([1994])
    torch.from_numpy(rewards) looks like  torch.Size([1994])
    rewards looks like  (1118,)
    log_probs looks like  (1118,)
    logs prob looks like  torch.Size([1118])
    torch.from_numpy(rewards) looks like  torch.Size([1118])
    rewards looks like  (1298,)
    log_probs looks like  (1298,)
    logs prob looks like  torch.Size([1298])
    torch.from_numpy(rewards) looks like  torch.Size([1298])
    rewards looks like  (1377,)
    log_probs looks like  (1377,)
    logs prob looks like  torch.Size([1377])
    torch.from_numpy(rewards) looks like  torch.Size([1377])
    rewards looks like  (1902,)
    log_probs looks like  (1902,)
    logs prob looks like  torch.Size([1902])
    torch.from_numpy(rewards) looks like  torch.Size([1902])
    rewards looks like  (1982,)
    log_probs looks like  (1982,)
    logs prob looks like  torch.Size([1982])
    torch.from_numpy(rewards) looks like  torch.Size([1982])
    rewards looks like  (1625,)
    log_probs looks like  (1625,)
    logs prob looks like  torch.Size([1625])
    torch.from_numpy(rewards) looks like  torch.Size([1625])
    rewards looks like  (1947,)
    log_probs looks like  (1947,)
    logs prob looks like  torch.Size([1947])
    torch.from_numpy(rewards) looks like  torch.Size([1947])
    rewards looks like  (1589,)
    log_probs looks like  (1589,)
    logs prob looks like  torch.Size([1589])
    torch.from_numpy(rewards) looks like  torch.Size([1589])
    rewards looks like  (1625,)
    log_probs looks like  (1625,)
    logs prob looks like  torch.Size([1625])
    torch.from_numpy(rewards) looks like  torch.Size([1625])
    rewards looks like  (1492,)
    log_probs looks like  (1492,)
    logs prob looks like  torch.Size([1492])
    torch.from_numpy(rewards) looks like  torch.Size([1492])
    rewards looks like  (1347,)
    log_probs looks like  (1347,)
    logs prob looks like  torch.Size([1347])
    torch.from_numpy(rewards) looks like  torch.Size([1347])
    rewards looks like  (2110,)
    log_probs looks like  (2110,)
    logs prob looks like  torch.Size([2110])
    torch.from_numpy(rewards) looks like  torch.Size([2110])
    rewards looks like  (877,)
    log_probs looks like  (877,)
    logs prob looks like  torch.Size([877])
    torch.from_numpy(rewards) looks like  torch.Size([877])
    rewards looks like  (1078,)
    log_probs looks like  (1078,)
    logs prob looks like  torch.Size([1078])
    torch.from_numpy(rewards) looks like  torch.Size([1078])
    rewards looks like  (2001,)
    log_probs looks like  (2001,)
    logs prob looks like  torch.Size([2001])
    torch.from_numpy(rewards) looks like  torch.Size([2001])
    rewards looks like  (1452,)
    log_probs looks like  (1452,)
    logs prob looks like  torch.Size([1452])
    torch.from_numpy(rewards) looks like  torch.Size([1452])
    rewards looks like  (1169,)
    log_probs looks like  (1169,)
    logs prob looks like  torch.Size([1169])
    torch.from_numpy(rewards) looks like  torch.Size([1169])
    rewards looks like  (1977,)
    log_probs looks like  (1977,)
    logs prob looks like  torch.Size([1977])
    torch.from_numpy(rewards) looks like  torch.Size([1977])
    rewards looks like  (1263,)
    log_probs looks like  (1263,)
    logs prob looks like  torch.Size([1263])
    torch.from_numpy(rewards) looks like  torch.Size([1263])
    rewards looks like  (2219,)
    log_probs looks like  (2219,)
    logs prob looks like  torch.Size([2219])
    torch.from_numpy(rewards) looks like  torch.Size([2219])
    rewards looks like  (1732,)
    log_probs looks like  (1732,)
    logs prob looks like  torch.Size([1732])
    torch.from_numpy(rewards) looks like  torch.Size([1732])
    rewards looks like  (1413,)
    log_probs looks like  (1413,)
    logs prob looks like  torch.Size([1413])
    torch.from_numpy(rewards) looks like  torch.Size([1413])
    rewards looks like  (1099,)
    log_probs looks like  (1099,)
    logs prob looks like  torch.Size([1099])
    torch.from_numpy(rewards) looks like  torch.Size([1099])
    rewards looks like  (1184,)
    log_probs looks like  (1184,)
    logs prob looks like  torch.Size([1184])
    torch.from_numpy(rewards) looks like  torch.Size([1184])
    rewards looks like  (1148,)
    log_probs looks like  (1148,)
    logs prob looks like  torch.Size([1148])
    torch.from_numpy(rewards) looks like  torch.Size([1148])
    rewards looks like  (1339,)
    log_probs looks like  (1339,)
    logs prob looks like  torch.Size([1339])
    torch.from_numpy(rewards) looks like  torch.Size([1339])
    rewards looks like  (2095,)
    log_probs looks like  (2095,)
    logs prob looks like  torch.Size([2095])
    torch.from_numpy(rewards) looks like  torch.Size([2095])
    rewards looks like  (1514,)
    log_probs looks like  (1514,)
    logs prob looks like  torch.Size([1514])
    torch.from_numpy(rewards) looks like  torch.Size([1514])
    rewards looks like  (1276,)
    log_probs looks like  (1276,)
    logs prob looks like  torch.Size([1276])
    torch.from_numpy(rewards) looks like  torch.Size([1276])
    rewards looks like  (1277,)
    log_probs looks like  (1277,)
    logs prob looks like  torch.Size([1277])
    torch.from_numpy(rewards) looks like  torch.Size([1277])
    rewards looks like  (1453,)
    log_probs looks like  (1453,)
    logs prob looks like  torch.Size([1453])
    torch.from_numpy(rewards) looks like  torch.Size([1453])
    rewards looks like  (1467,)
    log_probs looks like  (1467,)
    logs prob looks like  torch.Size([1467])
    torch.from_numpy(rewards) looks like  torch.Size([1467])
    rewards looks like  (1383,)
    log_probs looks like  (1383,)
    logs prob looks like  torch.Size([1383])
    torch.from_numpy(rewards) looks like  torch.Size([1383])
    rewards looks like  (1741,)
    log_probs looks like  (1741,)
    logs prob looks like  torch.Size([1741])
    torch.from_numpy(rewards) looks like  torch.Size([1741])
    rewards looks like  (1039,)
    log_probs looks like  (1039,)
    logs prob looks like  torch.Size([1039])
    torch.from_numpy(rewards) looks like  torch.Size([1039])
    rewards looks like  (1063,)
    log_probs looks like  (1063,)
    logs prob looks like  torch.Size([1063])
    torch.from_numpy(rewards) looks like  torch.Size([1063])
    rewards looks like  (1731,)
    log_probs looks like  (1731,)
    logs prob looks like  torch.Size([1731])
    torch.from_numpy(rewards) looks like  torch.Size([1731])
    rewards looks like  (2661,)
    log_probs looks like  (2661,)
    logs prob looks like  torch.Size([2661])
    torch.from_numpy(rewards) looks like  torch.Size([2661])
    rewards looks like  (704,)
    log_probs looks like  (704,)
    logs prob looks like  torch.Size([704])
    torch.from_numpy(rewards) looks like  torch.Size([704])
    rewards looks like  (1389,)
    log_probs looks like  (1389,)
    logs prob looks like  torch.Size([1389])
    torch.from_numpy(rewards) looks like  torch.Size([1389])
    rewards looks like  (2131,)
    log_probs looks like  (2131,)
    logs prob looks like  torch.Size([2131])
    torch.from_numpy(rewards) looks like  torch.Size([2131])
    rewards looks like  (1779,)
    log_probs looks like  (1779,)
    logs prob looks like  torch.Size([1779])
    torch.from_numpy(rewards) looks like  torch.Size([1779])
    rewards looks like  (1415,)
    log_probs looks like  (1415,)
    logs prob looks like  torch.Size([1415])
    torch.from_numpy(rewards) looks like  torch.Size([1415])
    rewards looks like  (2320,)
    log_probs looks like  (2320,)
    logs prob looks like  torch.Size([2320])
    torch.from_numpy(rewards) looks like  torch.Size([2320])
    rewards looks like  (1147,)
    log_probs looks like  (1147,)
    logs prob looks like  torch.Size([1147])
    torch.from_numpy(rewards) looks like  torch.Size([1147])
    rewards looks like  (1022,)
    log_probs looks like  (1022,)
    logs prob looks like  torch.Size([1022])
    torch.from_numpy(rewards) looks like  torch.Size([1022])
    rewards looks like  (2141,)
    log_probs looks like  (2141,)
    logs prob looks like  torch.Size([2141])
    torch.from_numpy(rewards) looks like  torch.Size([2141])
    rewards looks like  (1362,)
    log_probs looks like  (1362,)
    logs prob looks like  torch.Size([1362])
    torch.from_numpy(rewards) looks like  torch.Size([1362])
    rewards looks like  (1450,)
    log_probs looks like  (1450,)
    logs prob looks like  torch.Size([1450])
    torch.from_numpy(rewards) looks like  torch.Size([1450])
    rewards looks like  (1546,)
    log_probs looks like  (1546,)
    logs prob looks like  torch.Size([1546])
    torch.from_numpy(rewards) looks like  torch.Size([1546])
    rewards looks like  (1166,)
    log_probs looks like  (1166,)
    logs prob looks like  torch.Size([1166])
    torch.from_numpy(rewards) looks like  torch.Size([1166])
    rewards looks like  (1647,)
    log_probs looks like  (1647,)
    logs prob looks like  torch.Size([1647])
    torch.from_numpy(rewards) looks like  torch.Size([1647])
    rewards looks like  (1205,)
    log_probs looks like  (1205,)
    logs prob looks like  torch.Size([1205])
    torch.from_numpy(rewards) looks like  torch.Size([1205])
    rewards looks like  (2098,)
    log_probs looks like  (2098,)
    logs prob looks like  torch.Size([2098])
    torch.from_numpy(rewards) looks like  torch.Size([2098])
    rewards looks like  (1940,)
    log_probs looks like  (1940,)
    logs prob looks like  torch.Size([1940])
    torch.from_numpy(rewards) looks like  torch.Size([1940])
    rewards looks like  (2191,)
    log_probs looks like  (2191,)
    logs prob looks like  torch.Size([2191])
    torch.from_numpy(rewards) looks like  torch.Size([2191])
    rewards looks like  (2740,)
    log_probs looks like  (2740,)
    logs prob looks like  torch.Size([2740])
    torch.from_numpy(rewards) looks like  torch.Size([2740])
    rewards looks like  (587,)
    log_probs looks like  (587,)
    logs prob looks like  torch.Size([587])
    torch.from_numpy(rewards) looks like  torch.Size([587])
    rewards looks like  (1063,)
    log_probs looks like  (1063,)
    logs prob looks like  torch.Size([1063])
    torch.from_numpy(rewards) looks like  torch.Size([1063])
    rewards looks like  (861,)
    log_probs looks like  (861,)
    logs prob looks like  torch.Size([861])
    torch.from_numpy(rewards) looks like  torch.Size([861])
    rewards looks like  (1051,)
    log_probs looks like  (1051,)
    logs prob looks like  torch.Size([1051])
    torch.from_numpy(rewards) looks like  torch.Size([1051])
    rewards looks like  (1389,)
    log_probs looks like  (1389,)
    logs prob looks like  torch.Size([1389])
    torch.from_numpy(rewards) looks like  torch.Size([1389])
    rewards looks like  (1152,)
    log_probs looks like  (1152,)
    logs prob looks like  torch.Size([1152])
    torch.from_numpy(rewards) looks like  torch.Size([1152])
    rewards looks like  (1103,)
    log_probs looks like  (1103,)
    logs prob looks like  torch.Size([1103])
    torch.from_numpy(rewards) looks like  torch.Size([1103])
    rewards looks like  (1887,)
    log_probs looks like  (1887,)
    logs prob looks like  torch.Size([1887])
    torch.from_numpy(rewards) looks like  torch.Size([1887])
    rewards looks like  (1753,)
    log_probs looks like  (1753,)
    logs prob looks like  torch.Size([1753])
    torch.from_numpy(rewards) looks like  torch.Size([1753])
    rewards looks like  (1372,)
    log_probs looks like  (1372,)
    logs prob looks like  torch.Size([1372])
    torch.from_numpy(rewards) looks like  torch.Size([1372])
    rewards looks like  (1056,)
    log_probs looks like  (1056,)
    logs prob looks like  torch.Size([1056])
    torch.from_numpy(rewards) looks like  torch.Size([1056])
    rewards looks like  (1465,)
    log_probs looks like  (1465,)
    logs prob looks like  torch.Size([1465])
    torch.from_numpy(rewards) looks like  torch.Size([1465])
    rewards looks like  (3297,)
    log_probs looks like  (3297,)
    logs prob looks like  torch.Size([3297])
    torch.from_numpy(rewards) looks like  torch.Size([3297])
    rewards looks like  (2492,)
    log_probs looks like  (2492,)
    logs prob looks like  torch.Size([2492])
    torch.from_numpy(rewards) looks like  torch.Size([2492])
    rewards looks like  (1580,)
    log_probs looks like  (1580,)
    logs prob looks like  torch.Size([1580])
    torch.from_numpy(rewards) looks like  torch.Size([1580])
    rewards looks like  (1357,)
    log_probs looks like  (1357,)
    logs prob looks like  torch.Size([1357])
    torch.from_numpy(rewards) looks like  torch.Size([1357])
    rewards looks like  (1227,)
    log_probs looks like  (1227,)
    logs prob looks like  torch.Size([1227])
    torch.from_numpy(rewards) looks like  torch.Size([1227])
    rewards looks like  (2123,)
    log_probs looks like  (2123,)
    logs prob looks like  torch.Size([2123])
    torch.from_numpy(rewards) looks like  torch.Size([2123])
    rewards looks like  (1864,)
    log_probs looks like  (1864,)
    logs prob looks like  torch.Size([1864])
    torch.from_numpy(rewards) looks like  torch.Size([1864])
    rewards looks like  (1324,)
    log_probs looks like  (1324,)
    logs prob looks like  torch.Size([1324])
    torch.from_numpy(rewards) looks like  torch.Size([1324])
    rewards looks like  (1281,)
    log_probs looks like  (1281,)
    logs prob looks like  torch.Size([1281])
    torch.from_numpy(rewards) looks like  torch.Size([1281])
    rewards looks like  (1366,)
    log_probs looks like  (1366,)
    logs prob looks like  torch.Size([1366])
    torch.from_numpy(rewards) looks like  torch.Size([1366])
    rewards looks like  (957,)
    log_probs looks like  (957,)
    logs prob looks like  torch.Size([957])
    torch.from_numpy(rewards) looks like  torch.Size([957])
    rewards looks like  (1187,)
    log_probs looks like  (1187,)
    logs prob looks like  torch.Size([1187])
    torch.from_numpy(rewards) looks like  torch.Size([1187])
    rewards looks like  (1625,)
    log_probs looks like  (1625,)
    logs prob looks like  torch.Size([1625])
    torch.from_numpy(rewards) looks like  torch.Size([1625])
    rewards looks like  (1605,)
    log_probs looks like  (1605,)
    logs prob looks like  torch.Size([1605])
    torch.from_numpy(rewards) looks like  torch.Size([1605])
    rewards looks like  (1015,)
    log_probs looks like  (1015,)
    logs prob looks like  torch.Size([1015])
    torch.from_numpy(rewards) looks like  torch.Size([1015])
    rewards looks like  (1565,)
    log_probs looks like  (1565,)
    logs prob looks like  torch.Size([1565])
    torch.from_numpy(rewards) looks like  torch.Size([1565])
    rewards looks like  (1353,)
    log_probs looks like  (1353,)
    logs prob looks like  torch.Size([1353])
    torch.from_numpy(rewards) looks like  torch.Size([1353])
    rewards looks like  (1321,)
    log_probs looks like  (1321,)
    logs prob looks like  torch.Size([1321])
    torch.from_numpy(rewards) looks like  torch.Size([1321])
    rewards looks like  (1074,)
    log_probs looks like  (1074,)
    logs prob looks like  torch.Size([1074])
    torch.from_numpy(rewards) looks like  torch.Size([1074])
    rewards looks like  (1301,)
    log_probs looks like  (1301,)
    logs prob looks like  torch.Size([1301])
    torch.from_numpy(rewards) looks like  torch.Size([1301])
    rewards looks like  (2105,)
    log_probs looks like  (2105,)
    logs prob looks like  torch.Size([2105])
    torch.from_numpy(rewards) looks like  torch.Size([2105])
    rewards looks like  (2008,)
    log_probs looks like  (2008,)
    logs prob looks like  torch.Size([2008])
    torch.from_numpy(rewards) looks like  torch.Size([2008])
    rewards looks like  (1885,)
    log_probs looks like  (1885,)
    logs prob looks like  torch.Size([1885])
    torch.from_numpy(rewards) looks like  torch.Size([1885])
    rewards looks like  (1184,)
    log_probs looks like  (1184,)
    logs prob looks like  torch.Size([1184])
    torch.from_numpy(rewards) looks like  torch.Size([1184])
    rewards looks like  (2551,)
    log_probs looks like  (2551,)
    logs prob looks like  torch.Size([2551])
    torch.from_numpy(rewards) looks like  torch.Size([2551])
    rewards looks like  (1330,)
    log_probs looks like  (1330,)
    logs prob looks like  torch.Size([1330])
    torch.from_numpy(rewards) looks like  torch.Size([1330])
    rewards looks like  (1510,)
    log_probs looks like  (1510,)
    logs prob looks like  torch.Size([1510])
    torch.from_numpy(rewards) looks like  torch.Size([1510])
    rewards looks like  (1330,)
    log_probs looks like  (1330,)
    logs prob looks like  torch.Size([1330])
    torch.from_numpy(rewards) looks like  torch.Size([1330])
    rewards looks like  (2157,)
    log_probs looks like  (2157,)
    logs prob looks like  torch.Size([2157])
    torch.from_numpy(rewards) looks like  torch.Size([2157])
    rewards looks like  (1276,)
    log_probs looks like  (1276,)
    logs prob looks like  torch.Size([1276])
    torch.from_numpy(rewards) looks like  torch.Size([1276])
    rewards looks like  (1188,)
    log_probs looks like  (1188,)
    logs prob looks like  torch.Size([1188])
    torch.from_numpy(rewards) looks like  torch.Size([1188])
    rewards looks like  (2381,)
    log_probs looks like  (2381,)
    logs prob looks like  torch.Size([2381])
    torch.from_numpy(rewards) looks like  torch.Size([2381])
    rewards looks like  (1450,)
    log_probs looks like  (1450,)
    logs prob looks like  torch.Size([1450])
    torch.from_numpy(rewards) looks like  torch.Size([1450])
    rewards looks like  (1612,)
    log_probs looks like  (1612,)
    logs prob looks like  torch.Size([1612])
    torch.from_numpy(rewards) looks like  torch.Size([1612])
    rewards looks like  (1780,)
    log_probs looks like  (1780,)
    logs prob looks like  torch.Size([1780])
    torch.from_numpy(rewards) looks like  torch.Size([1780])
    rewards looks like  (1350,)
    log_probs looks like  (1350,)
    logs prob looks like  torch.Size([1350])
    torch.from_numpy(rewards) looks like  torch.Size([1350])
    rewards looks like  (1459,)
    log_probs looks like  (1459,)
    logs prob looks like  torch.Size([1459])
    torch.from_numpy(rewards) looks like  torch.Size([1459])
    rewards looks like  (1958,)
    log_probs looks like  (1958,)
    logs prob looks like  torch.Size([1958])
    torch.from_numpy(rewards) looks like  torch.Size([1958])
    rewards looks like  (1325,)
    log_probs looks like  (1325,)
    logs prob looks like  torch.Size([1325])
    torch.from_numpy(rewards) looks like  torch.Size([1325])
    rewards looks like  (2168,)
    log_probs looks like  (2168,)
    logs prob looks like  torch.Size([2168])
    torch.from_numpy(rewards) looks like  torch.Size([2168])
    rewards looks like  (1682,)
    log_probs looks like  (1682,)
    logs prob looks like  torch.Size([1682])
    torch.from_numpy(rewards) looks like  torch.Size([1682])
    rewards looks like  (852,)
    log_probs looks like  (852,)
    logs prob looks like  torch.Size([852])
    torch.from_numpy(rewards) looks like  torch.Size([852])
    rewards looks like  (1757,)
    log_probs looks like  (1757,)
    logs prob looks like  torch.Size([1757])
    torch.from_numpy(rewards) looks like  torch.Size([1757])
    rewards looks like  (2313,)
    log_probs looks like  (2313,)
    logs prob looks like  torch.Size([2313])
    torch.from_numpy(rewards) looks like  torch.Size([2313])
    rewards looks like  (1662,)
    log_probs looks like  (1662,)
    logs prob looks like  torch.Size([1662])
    torch.from_numpy(rewards) looks like  torch.Size([1662])
    rewards looks like  (1559,)
    log_probs looks like  (1559,)
    logs prob looks like  torch.Size([1559])
    torch.from_numpy(rewards) looks like  torch.Size([1559])
    rewards looks like  (2077,)
    log_probs looks like  (2077,)
    logs prob looks like  torch.Size([2077])
    torch.from_numpy(rewards) looks like  torch.Size([2077])
    rewards looks like  (2119,)
    log_probs looks like  (2119,)
    logs prob looks like  torch.Size([2119])
    torch.from_numpy(rewards) looks like  torch.Size([2119])
    rewards looks like  (954,)
    log_probs looks like  (954,)
    logs prob looks like  torch.Size([954])
    torch.from_numpy(rewards) looks like  torch.Size([954])
    rewards looks like  (1797,)
    log_probs looks like  (1797,)
    logs prob looks like  torch.Size([1797])
    torch.from_numpy(rewards) looks like  torch.Size([1797])
    rewards looks like  (1579,)
    log_probs looks like  (1579,)
    logs prob looks like  torch.Size([1579])
    torch.from_numpy(rewards) looks like  torch.Size([1579])
    rewards looks like  (1277,)
    log_probs looks like  (1277,)
    logs prob looks like  torch.Size([1277])
    torch.from_numpy(rewards) looks like  torch.Size([1277])
    rewards looks like  (1196,)
    log_probs looks like  (1196,)
    logs prob looks like  torch.Size([1196])
    torch.from_numpy(rewards) looks like  torch.Size([1196])
    rewards looks like  (1294,)
    log_probs looks like  (1294,)
    logs prob looks like  torch.Size([1294])
    torch.from_numpy(rewards) looks like  torch.Size([1294])
    rewards looks like  (1318,)
    log_probs looks like  (1318,)
    logs prob looks like  torch.Size([1318])
    torch.from_numpy(rewards) looks like  torch.Size([1318])
    rewards looks like  (2605,)
    log_probs looks like  (2605,)
    logs prob looks like  torch.Size([2605])
    torch.from_numpy(rewards) looks like  torch.Size([2605])
    rewards looks like  (2002,)
    log_probs looks like  (2002,)
    logs prob looks like  torch.Size([2002])
    torch.from_numpy(rewards) looks like  torch.Size([2002])
    rewards looks like  (1354,)
    log_probs looks like  (1354,)
    logs prob looks like  torch.Size([1354])
    torch.from_numpy(rewards) looks like  torch.Size([1354])
    rewards looks like  (1785,)
    log_probs looks like  (1785,)
    logs prob looks like  torch.Size([1785])
    torch.from_numpy(rewards) looks like  torch.Size([1785])
    rewards looks like  (781,)
    log_probs looks like  (781,)
    logs prob looks like  torch.Size([781])
    torch.from_numpy(rewards) looks like  torch.Size([781])
    rewards looks like  (1965,)
    log_probs looks like  (1965,)
    logs prob looks like  torch.Size([1965])
    torch.from_numpy(rewards) looks like  torch.Size([1965])
    rewards looks like  (1135,)
    log_probs looks like  (1135,)
    logs prob looks like  torch.Size([1135])
    torch.from_numpy(rewards) looks like  torch.Size([1135])
    rewards looks like  (1672,)
    log_probs looks like  (1672,)
    logs prob looks like  torch.Size([1672])
    torch.from_numpy(rewards) looks like  torch.Size([1672])
    rewards looks like  (1278,)
    log_probs looks like  (1278,)
    logs prob looks like  torch.Size([1278])
    torch.from_numpy(rewards) looks like  torch.Size([1278])
    rewards looks like  (2499,)
    log_probs looks like  (2499,)
    logs prob looks like  torch.Size([2499])
    torch.from_numpy(rewards) looks like  torch.Size([2499])
    rewards looks like  (1275,)
    log_probs looks like  (1275,)
    logs prob looks like  torch.Size([1275])
    torch.from_numpy(rewards) looks like  torch.Size([1275])
    rewards looks like  (1144,)
    log_probs looks like  (1144,)
    logs prob looks like  torch.Size([1144])
    torch.from_numpy(rewards) looks like  torch.Size([1144])
    rewards looks like  (1605,)
    log_probs looks like  (1605,)
    logs prob looks like  torch.Size([1605])
    torch.from_numpy(rewards) looks like  torch.Size([1605])
    rewards looks like  (1178,)
    log_probs looks like  (1178,)
    logs prob looks like  torch.Size([1178])
    torch.from_numpy(rewards) looks like  torch.Size([1178])
    rewards looks like  (3269,)
    log_probs looks like  (3269,)
    logs prob looks like  torch.Size([3269])
    torch.from_numpy(rewards) looks like  torch.Size([3269])
    rewards looks like  (1492,)
    log_probs looks like  (1492,)
    logs prob looks like  torch.Size([1492])
    torch.from_numpy(rewards) looks like  torch.Size([1492])
    rewards looks like  (1285,)
    log_probs looks like  (1285,)
    logs prob looks like  torch.Size([1285])
    torch.from_numpy(rewards) looks like  torch.Size([1285])
    rewards looks like  (1687,)
    log_probs looks like  (1687,)
    logs prob looks like  torch.Size([1687])
    torch.from_numpy(rewards) looks like  torch.Size([1687])
    rewards looks like  (1124,)
    log_probs looks like  (1124,)
    logs prob looks like  torch.Size([1124])
    torch.from_numpy(rewards) looks like  torch.Size([1124])
    rewards looks like  (2043,)
    log_probs looks like  (2043,)
    logs prob looks like  torch.Size([2043])
    torch.from_numpy(rewards) looks like  torch.Size([2043])
    rewards looks like  (1280,)
    log_probs looks like  (1280,)
    logs prob looks like  torch.Size([1280])
    torch.from_numpy(rewards) looks like  torch.Size([1280])
    rewards looks like  (1418,)
    log_probs looks like  (1418,)
    logs prob looks like  torch.Size([1418])
    torch.from_numpy(rewards) looks like  torch.Size([1418])
    rewards looks like  (1365,)
    log_probs looks like  (1365,)
    logs prob looks like  torch.Size([1365])
    torch.from_numpy(rewards) looks like  torch.Size([1365])
    rewards looks like  (1091,)
    log_probs looks like  (1091,)
    logs prob looks like  torch.Size([1091])
    torch.from_numpy(rewards) looks like  torch.Size([1091])
    rewards looks like  (1279,)
    log_probs looks like  (1279,)
    logs prob looks like  torch.Size([1279])
    torch.from_numpy(rewards) looks like  torch.Size([1279])
    rewards looks like  (1109,)
    log_probs looks like  (1109,)
    logs prob looks like  torch.Size([1109])
    torch.from_numpy(rewards) looks like  torch.Size([1109])
    rewards looks like  (1285,)
    log_probs looks like  (1285,)
    logs prob looks like  torch.Size([1285])
    torch.from_numpy(rewards) looks like  torch.Size([1285])
    rewards looks like  (1222,)
    log_probs looks like  (1222,)
    logs prob looks like  torch.Size([1222])
    torch.from_numpy(rewards) looks like  torch.Size([1222])
    rewards looks like  (1538,)
    log_probs looks like  (1538,)
    logs prob looks like  torch.Size([1538])
    torch.from_numpy(rewards) looks like  torch.Size([1538])
    rewards looks like  (1139,)
    log_probs looks like  (1139,)
    logs prob looks like  torch.Size([1139])
    torch.from_numpy(rewards) looks like  torch.Size([1139])
    rewards looks like  (1354,)
    log_probs looks like  (1354,)
    logs prob looks like  torch.Size([1354])
    torch.from_numpy(rewards) looks like  torch.Size([1354])
    rewards looks like  (1166,)
    log_probs looks like  (1166,)
    logs prob looks like  torch.Size([1166])
    torch.from_numpy(rewards) looks like  torch.Size([1166])
    rewards looks like  (1348,)
    log_probs looks like  (1348,)
    logs prob looks like  torch.Size([1348])
    torch.from_numpy(rewards) looks like  torch.Size([1348])
    rewards looks like  (1347,)
    log_probs looks like  (1347,)
    logs prob looks like  torch.Size([1347])
    torch.from_numpy(rewards) looks like  torch.Size([1347])
    rewards looks like  (2059,)
    log_probs looks like  (2059,)
    logs prob looks like  torch.Size([2059])
    torch.from_numpy(rewards) looks like  torch.Size([2059])
    rewards looks like  (2021,)
    log_probs looks like  (2021,)
    logs prob looks like  torch.Size([2021])
    torch.from_numpy(rewards) looks like  torch.Size([2021])
    rewards looks like  (2232,)
    log_probs looks like  (2232,)
    logs prob looks like  torch.Size([2232])
    torch.from_numpy(rewards) looks like  torch.Size([2232])
    rewards looks like  (1102,)
    log_probs looks like  (1102,)
    logs prob looks like  torch.Size([1102])
    torch.from_numpy(rewards) looks like  torch.Size([1102])
    rewards looks like  (1165,)
    log_probs looks like  (1165,)
    logs prob looks like  torch.Size([1165])
    torch.from_numpy(rewards) looks like  torch.Size([1165])
    rewards looks like  (1264,)
    log_probs looks like  (1264,)
    logs prob looks like  torch.Size([1264])
    torch.from_numpy(rewards) looks like  torch.Size([1264])
    rewards looks like  (1346,)
    log_probs looks like  (1346,)
    logs prob looks like  torch.Size([1346])
    torch.from_numpy(rewards) looks like  torch.Size([1346])
    rewards looks like  (2848,)
    log_probs looks like  (2848,)
    logs prob looks like  torch.Size([2848])
    torch.from_numpy(rewards) looks like  torch.Size([2848])
    rewards looks like  (938,)
    log_probs looks like  (938,)
    logs prob looks like  torch.Size([938])
    torch.from_numpy(rewards) looks like  torch.Size([938])
    rewards looks like  (1069,)
    log_probs looks like  (1069,)
    logs prob looks like  torch.Size([1069])
    torch.from_numpy(rewards) looks like  torch.Size([1069])
    rewards looks like  (2588,)
    log_probs looks like  (2588,)
    logs prob looks like  torch.Size([2588])
    torch.from_numpy(rewards) looks like  torch.Size([2588])
    rewards looks like  (1461,)
    log_probs looks like  (1461,)
    logs prob looks like  torch.Size([1461])
    torch.from_numpy(rewards) looks like  torch.Size([1461])
    rewards looks like  (2153,)
    log_probs looks like  (2153,)
    logs prob looks like  torch.Size([2153])
    torch.from_numpy(rewards) looks like  torch.Size([2153])
    rewards looks like  (2312,)
    log_probs looks like  (2312,)
    logs prob looks like  torch.Size([2312])
    torch.from_numpy(rewards) looks like  torch.Size([2312])
    rewards looks like  (1636,)
    log_probs looks like  (1636,)
    logs prob looks like  torch.Size([1636])
    torch.from_numpy(rewards) looks like  torch.Size([1636])
    rewards looks like  (2019,)
    log_probs looks like  (2019,)
    logs prob looks like  torch.Size([2019])
    torch.from_numpy(rewards) looks like  torch.Size([2019])
    rewards looks like  (1450,)
    log_probs looks like  (1450,)
    logs prob looks like  torch.Size([1450])
    torch.from_numpy(rewards) looks like  torch.Size([1450])
    rewards looks like  (2105,)
    log_probs looks like  (2105,)
    logs prob looks like  torch.Size([2105])
    torch.from_numpy(rewards) looks like  torch.Size([2105])
    


### Training Result
During the training process, we recorded `avg_total_reward`, which represents the average total reward of episodes before updating the policy network.

Theoretically, if the agent becomes better, the `avg_total_reward` will increase.
The visualization of the training process is shown below:  



```
plt.plot(avg_total_rewards)
plt.title("Total Rewards")
plt.show()
```


    
![png](hw12_reinforcement_learning_english_version_files/hw12_reinforcement_learning_english_version_34_0.png)
    


In addition, `avg_final_reward` represents average final rewards of episodes. To be specific, final rewards is the last reward received in one episode, indicating whether the craft lands successfully or not.



```
plt.plot(avg_final_rewards)
plt.title("Final Rewards")
plt.show()
```


    
![png](hw12_reinforcement_learning_english_version_files/hw12_reinforcement_learning_english_version_36_0.png)
    


## Testing
The testing result will be the average reward of 5 testing


```
fix(env, seed)
agent.network.eval()  # set the network into evaluation mode
NUM_OF_TEST = 5 # Do not revise this !!!
test_total_reward = []
action_list = []
for i in range(NUM_OF_TEST):
  actions = []
  state = env.reset()

  img = plt.imshow(env.render(mode='rgb_array'))

  total_reward = 0

  done = False
  while not done:
      action, _ = agent.sample(state)
      actions.append(action)
      state, reward, done, _ = env.step(action)

      total_reward += reward

      img.set_data(env.render(mode='rgb_array'))
      display.display(plt.gcf())
      display.clear_output(wait=True)
      
  print(total_reward)
  test_total_reward.append(total_reward)

  action_list.append(actions) # save the result of testing 

```

    -207.9114975585693



    
![png](hw12_reinforcement_learning_english_version_files/hw12_reinforcement_learning_english_version_38_1.png)
    



```
print(np.mean(test_total_reward))
```

    -147.2620449863271


Action list


```
print("Action list looks like ", action_list)
print("Action list's shape looks like ", np.shape(action_list))
```

    Action list looks like  [[2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 0, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 2, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 0, 2, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 2, 2, 2, 3, 3, 2, 3, 0, 2, 3, 2, 0, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 0, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 0, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 2, 3, 3, 3, 2, 3, 3, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 0, 1, 2, 2, 0, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 0, 2, 0, 3, 2, 3, 2, 0, 2, 0, 2, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 0, 2, 3, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 1, 2, 2, 0, 2, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 3, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 2, 2, 3, 2, 3, 2, 2, 2, 3, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 0, 2, 2, 1, 2, 2, 2, 3, 0, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 2, 3, 3, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 2, 1], [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 2, 3, 3, 2, 2, 3, 3, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 0, 2, 1, 2, 2, 0, 1, 2, 1, 2, 2, 1, 1, 2, 0, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 1, 0, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 3, 2, 3, 2, 2, 3, 2, 3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 2, 2, 2, 2, 3, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 2, 2, 2, 2, 1, 1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2, 0, 2, 2, 2, 3, 2, 3, 2, 2, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 2, 3, 3, 3, 3, 2, 3, 3, 2, 3, 3, 3, 2, 2, 3, 2, 3, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 3, 2, 3, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]]
    Action list's shape looks like  (5,)


    /usr/local/lib/python3.7/dist-packages/numpy/core/_asarray.py:83: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      return array(a, dtype, copy=False, order=order)


Analysis of actions taken by agent


```
distribution = {}
for actions in action_list:
  for action in actions:
    if action not in distribution.keys():
      distribution[action] = 1
    else:
      distribution[action] += 1
print(distribution)
```

    {2: 1144, 1: 516, 0: 30, 3: 501}


Saving the result of Model Testing



```
PATH = "Action_List.npy" # Can be modified into the name or path you want
np.save(PATH ,np.array(action_list)) 
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray
      


### This is the file you need to submit !!!
Download the testing result to your device




```
from google.colab import files
files.download(PATH)
```


    <IPython.core.display.Javascript object>



    <IPython.core.display.Javascript object>


# Server 
The code below simulate the environment on the judge server. Can be used for testing.


```
action_list = np.load(PATH,allow_pickle=True) # The action list you upload
seed = 543 # Do not revise this
fix(env, seed)

agent.network.eval()  # set network to evaluation mode

test_total_reward = []
if len(action_list) != 5:
  print("Wrong format of file !!!")
  exit(0)
for actions in action_list:
  state = env.reset()
  img = plt.imshow(env.render(mode='rgb_array'))

  total_reward = 0

  done = False

  for action in actions:
  
      state, reward, done, _ = env.step(action)
      total_reward += reward
      if done:
        break

  print(f"Your reward is : %.2f"%total_reward)
  test_total_reward.append(total_reward)
```

    /usr/local/lib/python3.7/dist-packages/torch/__init__.py:422: UserWarning: torch.set_deterministic is deprecated and will be removed in a future release. Please use torch.use_deterministic_algorithms instead
      "torch.set_deterministic is deprecated and will be removed in a future "


    Your reward is : -29.53
    Your reward is : -36.44
    Your reward is : -194.16
    Your reward is : -268.27
    Your reward is : -207.91



    
![png](hw12_reinforcement_learning_english_version_files/hw12_reinforcement_learning_english_version_49_2.png)
    


# Your score


```
print(f"Your final reward is : %.2f"%np.mean(test_total_reward))
```

    Your final reward is : -147.26


## Reference

Below are some useful tips for you to get high score.

- [DRL Lecture 1: Policy Gradient (Review)](https://youtu.be/z95ZYgPgXOY)
- [ML Lecture 23-3: Reinforcement Learning (including Q-learning) start at 30:00](https://youtu.be/2-JNBzCq77c?t=1800)
- [Lecture 7: Policy Gradient, David Silver](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching_files/pg.pdf)



