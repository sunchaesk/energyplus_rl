# installed tianshou
import base2 as base

import gymnasium as gym
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import Tensorboard
import tianshou as ts
from tianshou.utils.net.common import Net

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

# hyperparameters
lr, epoch, batch_size = 1e-3, 10, 64
train_num = 10, 100
gamma, n_step, target_freq = 0.99, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.1, 0.05

env = base.EnergyPlusEnv(default_args)
state_shape = env.observation_space.shape
action_shape = env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128,128,128])
