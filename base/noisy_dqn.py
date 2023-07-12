import base2 as base
import time
import sys
import os

import numpy as np
import random
from collections import namedtuple, deque
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

import matplotlib.pyplot as plt

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 500        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 1        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using: ", device)


class NoisyLinear(nn.Linear):
  # Noisy Linear Layer for independent Gaussian Noise
  def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
    super(NoisyLinear, self).__init__(in_features, out_features, bias = bias)
    # make the sigmas trainable:
    self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
    # not trainable tensor for the nn.Module
    self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
    # extra parameter for the bias and register buffer for the bias parameter
    if bias:
      self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
      self.register_buffer("epsilon_bias", torch.zeros(out_features))

    # reset parameter as initialization of the layer
    self.reset_parameter()

  def reset_parameter(self):
    """
    initialize the parameter of the layer and bias
    """
    std = math.sqrt(3/self.in_features)
    self.weight.data.uniform_(-std, std)
    self.bias.data.uniform_(-std, std)


  def forward(self, input):
    # sample random noise in sigma weight buffer and bias buffer
    self.epsilon_weight.normal_()
    bias = self.bias
    if bias is not None:
      self.epsilon_bias.normal_()
      bias = bias + self.sigma_bias * self.epsilon_bias
    return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)




class Dueling_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Dueling_QNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        self.noisy_layers = [NoisyLinear(state_size,fc1_units),
                             NoisyLinear(fc1_units,fc2_units),
                             NoisyLinear(fc2_units,action_size),
                             NoisyLinear(fc2_units,1)]

        self.network = nn.Sequential(self.noisy_layers[0],
                                     nn.ReLU(),
                                     self.noisy_layers[1],
                                     nn.ReLU())
        self.advantage = nn.Sequential(self.noisy_layers[2])
        self.value = nn.Sequential(self.noisy_layers[3])

    def forward(self, state):
        x = self.network(state)
        value = self.value(x)
        value = value.expand(x.size(0), self.action_size)
        advantage = self.advantage(x)
        Q = value + advantage - advantage.mean()
        return Q

    def noisy_layer_sigma_snr(self):
      """
      function to monitor Noise SNR (signal-to-noise ratio) RMS(mu)/RMS(sigma)
      """
      return [((layer.weight**2).mean().sqrt() / (layer.sigma_weight**2).mean().sqrt()).item() for layer in self.noisy_layers]



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

	    # Q-Network

        self.qnetwork_local = Dueling_QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = Dueling_QNetwork(state_size, action_size, seed).to(device)


        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        return np.argmax(action_values.cpu().data.numpy())


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def dqn(n_episodes=1000, max_t=2000):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode

    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0

        cooling_setpoints = []
        cost_signals = []
        outdoor_temperatures = []
        indoor_temperatures = []

        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.step(state, action, reward, next_state, done)

            cooling_setpoints.append(info['cooling_actuator_value'])
            cost_signals.append(info['cost_signal'])
            outdoor_temperatures.append(next_state[0])
            indoor_temperatures.append(next_state[1])

            state = next_state
            score += reward
            if done:
                break
        save_reward(score)
        if score < 4500:
          model_save('noisy_checkpoint', agent, i_episode)
          print('MODEL <4500 reached... Save current model')
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, i_episode)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 2 == 0 and i_episode != 0:
          model_save('noisy_checkpoint', agent, i_episode)
          print('MODEL SAVING! (%2)')
          time.sleep(2)

    return scores

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

def model_save(save_name, agent, episode):
  torch.save({
    'episode': episode,
    'q_local': agent.qnetwork_local.state_dict(),
    'q_target': agent.qnetwork_target.state_dict(),
    'optimizer': agent.optimizer.state_dict(),
    'memory': agent.memory
  }, './model/' + save_name + '.pt')

def graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, episode):
    start = 310
    end = 1010
    x = list(range(end - start))

    print(len(x))
    print(len(cooling_setpoints[start:end]))

    fig, ax1 = plt.subplots()
    ax1.set_title('10 - 310 steps of training {} episodes'.format(episode))
    ax1.scatter(x, cooling_setpoints[start:end], color='red')
    ax1.plot(x, outdoor_temperatures[start:end], linestyle='--', color='green')
    ax1.plot(x, indoor_temperatures[start:end], linestyle='--', color='magenta')

    ax2 = ax1.twinx()
    ax2.plot(x, cost_signals[start:end])
    fig.tight_layout()
    plt.savefig('./logs/noisy.png')

def save_reward(reward: float) -> None:
  f_name = './logs/noisy_scores.txt'
  with open(f_name, 'a') as scores_f:
    scores_f.write(str(reward) + '\n')


env = base.EnergyPlusEnv(default_args)
action_size = env.action_space.n
state_size = env.observation_space.shape[0]

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
load = True
try:
  checkpoint = torch.load('./model/noisy_checkpoint.pt')
  agent.qnetwork_local.load_state_dict(checkpoint['q_local'])
  agent.qnetwork_target.load_state_dict(checkpoint['q_target'])
  agent.optimizer.load_state_dict(checkpoint['optimizer'])
  agent.memory = checkpoint['memory']
except:
  print("ERROR: skipping loading model")
scores = dqn(n_episodes = 100000)

  # plot the scores
