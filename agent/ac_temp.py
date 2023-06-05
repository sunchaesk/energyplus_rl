
import base_cont as base

import gym
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm_
from collections import deque



class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.hidden_size = 128
        self.net = nn.Sequential(nn.Linear(input_shape, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size, 1))

    def forward(self,x):
        x = self.net(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Actor, self).__init__()
        self.hidden_size = 128
        self.net = nn.Sequential(nn.Linear(input_shape, self.hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden_size,self.hidden_size),
                                 nn.ReLU(),
                                 )
        self.mean = nn.Sequential(nn.Linear(self.hidden_size, output_shape),
                                  nn.Tanh())                    # tanh squashed output to the range of -1..1
        self.variance =nn.Sequential(nn.Linear(self.hidden_size, output_shape),
                                     nn.Softplus())             # log(1 + e^x) has the shape of a smoothed ReLU

    def forward(self,x):
        x = self.net(x)
        return self.mean(x), self.variance(x)

class Agent():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.95
        self.clip_grad = 0.1
        self.lr_a = 1e-3
        self.lr_c = 1e-3
        self.device = 'cpu'
        self.entropy_beta = 0.001

        self.input_shape = self.env.observation_space.shape[0]
        self.output_shape = self.env.action_space.shape[0]

        self.critic = Critic(self.input_shape).to(self.device)
        self.actor = Actor(self.input_shape, self.output_shape).to(self.device)

        self.c_optimizer = optim.Adam(parameters=self.critic.parameters, lr=self.lr_c)
        self.a_optimizer = optim.Adam(parameters=self.actor.parameters, lr= self.lr_a)


    def compute_returns(self,rewards,masks):
        R = 0 #pred.detach()
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return torch.FloatTensor(returns).reshape(-1).unsqueeze(1)


    def sample(self, mean, variance):
        """
        Calculates the actions, log probs and entropy based on a normal distribution by a given mean and variance.

        ====================================================

        calculate log prob:
        log_prob = -((action - mean) ** 2) / (2 * var) - log(sigma) - log(sqrt(2 *pi))

        calculate entropy:
        entropy =  0.5 + 0.5 * log(2 *pi * sigma)
        entropy here relates directly to the unpredictability of the actions which an agent takes in a given policy.
        The greater the entropy, the more random the actions an agent takes.

        """
        sigma = torch.sqrt(variance)
        m = Normal(mean, sigma)
        actions = m.sample()
        actions = torch.clamp(actions, -1, 1) # usually clipping between -1,1 but pendulum env has action range of -2,2
        logprobs = m.log_prob(actions)
        entropy = m.entropy()  # Equation: 0.5 + 0.5 * log(2 *pi * sigma)

        return actions, logprobs, entropy

    def run_optimization(self, logprob_batch, entropy_batch, values_batch, rewards_batch, masks):
        """
        Calculates the actor loss and the critic loss and backpropagates it through the Network

        ============================================
        Critic loss:
        c_loss = -logprob * advantage

        a_loss =

        """

        log_prob_v = torch.cat(logprob_batch).to(self.device)
        entropy_v = torch.cat(entropy_batch).to(self.device)
        value_v = torch.cat(values_batch).to(self.device)



        rewards_batch = torch.FloatTensor(rewards_batch)
        masks = torch.FloatTensor(masks)
        discounted_rewards = self.compute_returns(rewards_batch, masks).to(self.device)

        # critic_loss
        self.c_optimizer.zero_grad()
        critic_loss = 0.5 * F.mse_loss(value_v, discounted_rewards) #+ ENTROPY_BETA * entropy.detach().mean()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.c_optimizer.step()

        # A(s,a) = Q(s,a) - V(s)
        advantage = discounted_rewards - value_v.detach()

        #actor_loss
        self.a_optimizer.zero_grad()
        actor_loss = (-log_prob_v * advantage).mean() + self.entropy_beta * entropy.detach().mean()
        print('ENTROPY:', entropy)
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(),self.clip_grad)
        self.a_optimizer.step()

        return actor_loss, critic_loss
