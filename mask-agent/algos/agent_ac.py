
import base_test as base

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
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime
#import pybullet_envs
#env_name = "MountainCarContinuous-v0" #"MountainCarContinuous-v0"  #Pendulum-v0 LunarLanderContinuous-v2

writer = SummaryWriter()

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                'start_date': (6,21),
                'end_date': (8,21)
                }

env = base.EnergyPlusEnv(default_args)

print("action space: ", env.action_space.shape[0])
print("observation space ", env.observation_space.shape[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using: ", device)


GAMMA = 0.9
# GAMMA = 0.99
ENTROPY_BETA = 0.001
CLIP_GRAD = .1
LR_c = 1e-4
LR_a = 1e-4

HIDDEN_SIZE = 32
# HIDDEN_SIZE = 256

class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_shape, HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE, 1))

    def forward(self,x):
        x = self.net(x)
        return x

class Actor(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_shape, HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE),
                                 nn.ReLU(),
                                 )
        self.mean = nn.Sequential(nn.Linear(HIDDEN_SIZE, output_shape),
                                  nn.Tanh())                    # tanh squashed output to the range of -1..1
        self.variance =nn.Sequential(nn.Linear(HIDDEN_SIZE, output_shape),
                                     nn.Softplus())             # log(1 + e^x) has the shape of a smoothed ReLU

    def forward(self,x):
        x = self.net(x)
        return self.mean(x), self.variance(x)



# NOTE: for infinite horizon
def compute_returns(rewards,masks, gamma=GAMMA):
    '''
    infinite horizon
    '''
    R = 0 #pred.detach()
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return torch.FloatTensor(returns).reshape(-1).unsqueeze(1)

# NOTE: for finite horizon
# def compute_returns(rews, masks):
#     '''
#     rews: rewards
#     masks: masks but not used
#     '''
#     rews = list(reversed(rews))
#     ret = np.zeros_like(rews)
#     # horizon = 35  # set it to one less than desired T -> 6 * 6 - 1
#     horizon = 71
#     for i in range(len(rews)):
#         temp = 0
#         if i - horizon > 0:
#             end = i - horizon - 1
#         else:
#             end = -1
#         for j in range(i, end, -1):
#             temp += rews[j]
#         ret[i] = temp
#     ret = list(reversed(ret))
#     return torch.FloatTensor(ret).reshape(-1).unsqueeze(1)


def sample(mean, variance, mask_range):
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
    #print('ACTIONS: ', actions)
    actions = torch.clamp(actions, -1, 1) # usually clipping between -1,1 but pendulum env has action range of -2,2
    logprobs = m.log_prob(actions)
    entropy = m.entropy()  # Equation: 0.5 + 0.5 * log(2 *pi * sigma)
    #print('ENTROPY?', entropy)

    return actions, logprobs, entropy


def run_optimization(logprob_batch, entropy_batch, values_batch, rewards_batch, masks):
    """
    Calculates the actor loss and the critic loss and backpropagates it through the Network

    ============================================
    Critic loss:
    c_loss = -logprob * advantage

    a_loss =

    """

    log_prob_v = torch.cat(logprob_batch).to(device)
    entropy_v = torch.cat(entropy_batch).to(device)
    value_v = torch.cat(values_batch).to(device)



    rewards_batch = torch.FloatTensor(rewards_batch)
    masks = torch.FloatTensor(masks)
    discounted_rewards = compute_returns(rewards_batch, masks).to(device)

    # critic_loss
    c_optimizer.zero_grad()
    critic_loss = 0.5 * F.mse_loss(value_v, discounted_rewards) #+ ENTROPY_BETA * entropy.detach().mean()
    critic_loss.backward()
    clip_grad_norm_(critic.parameters(),CLIP_GRAD)
    c_optimizer.step()

    # A(s,a) = Q(s,a) - V(s)
    advantage = discounted_rewards - value_v.detach()

    #actor_loss
    a_optimizer.zero_grad()
    actor_loss = (-log_prob_v * advantage).mean() + ENTROPY_BETA * entropy.detach().mean()
    print('ENTROPY:', entropy)
    actor_loss.backward()
    clip_grad_norm_(actor.parameters(),CLIP_GRAD)
    a_optimizer.step()

    return actor_loss, critic_loss


f_name = './logs/scores'
# f_name = './logs/scores-'
# f_name += datetime.now().strftime('%m-%d-%H:%M')
f_name += '.txt'
def save_reward(score:float) -> None:
    with open(f_name, 'a') as scores_f:
        scores_f.write(str(score) + '\n')

f_name2 = './thermal_log.txt'
def save_thermal_comfort(score:float) -> None:
    with open(f_name2, 'a') as scores_f:
        scores_f.write(str(score) + '\n')

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)
# np.random.seed(42)
# env.seed(42)


input_shape  = env.observation_space.shape[0]
output_shape = env.action_space.shape[0]

critic = Critic(input_shape).to(device)
actor = Actor(input_shape, output_shape).to(device)
c_optimizer = optim.Adam(params = critic.parameters(),lr = LR_c)
a_optimizer = optim.Adam(params = actor.parameters(),lr = LR_a)

max_episodes = 20000

actor_loss_list = []
critic_loss_list = []
entropy_list = []


average_100 = []
plot_rewards = []
steps = 0

start_episode = 0
load = True
if load:
    try:
        checkpoint = torch.load('./model/checkpoint.pt')
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        c_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        a_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        start_episode = checkpoint['episode'] + 1
    except:
        print('ERROR: ./model/checkpoint.pt not found -> training from episode = 0')

for ep in range(start_episode, max_episodes + 1):
    state = env.reset()
    done = False

    episode_reward = 0

    logprob_batch = []
    entropy_batch = []
    values_batch = []
    rewards_batch = []
    masks = []

    # for logging
    #thermal_comfort_list = []

    while not done:

        state = torch.from_numpy(state).float()

        if env.b_during_sim():
            mean, variance = actor(state.unsqueeze(0).to(device))
            action, logprob, entropy = sample(mean.cpu(), variance.cpu(), tuple([-1,1]))
            value = critic(state.unsqueeze(0).to(device))
            next_state, reward, done, truncated, info = env.step(action[0].numpy())
            print('NEXT_STATE', next_state)
            steps += 1
            episode_reward += info['energy_reward']
            state = next_state

            # NOTE: for logging
            #save_thermal_comfort(info['comfort_reward'])
            #thermal_comfort_list.append(info['comfort_reward'])
        else:
            action = env.action_space.sample()
            #print('ACTION:', action)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            print('\r skipping... date', str(info['date']), end='', flush=True)
            continue


        logprob_batch.append(logprob)
        entropy_batch.append(entropy)
        values_batch.append(value)
        rewards_batch.append(reward)
        masks.append(1 - done)

        writer.add_scalar('LogProb/Steps', logprob, steps)
        writer.add_scalar('Entropy/Steps', entropy, steps)
        writer.add_scalar('Values/Steps', value, steps)
        writer.add_scalar('Rewards/Steps', reward, steps)

        state = next_state

        if done:
          break

    actor_loss, critic_loss = run_optimization(logprob_batch, entropy_batch, values_batch, rewards_batch, masks)

    actor_loss_list.append(actor_loss)
    critic_loss_list.append(critic_loss)

    print('################')
    print("\rEpisode: {} | Ep_Reward: {:.2f}".format(ep, episode_reward), end = "\n", flush = True)
    print('################')

    # save stuff
    save_reward(episode_reward)

    if ep != 0 and ep % 2 == 0:
        print('Saveing model episode:' + str(ep) + ' to ./model/checkpoint.pt')
        torch.save({
            'episode': ep,
            'actor_state_dict': actor.state_dict(),
            'critic_state_dict': critic.state_dict(),
            'actor_optimizer': a_optimizer.state_dict(),
            'critic_optimizer': c_optimizer.state_dict(),
        }, './model/checkpoint.pt')
