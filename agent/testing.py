
'''
pip installed mplcursors, mpld3
'''

import time
import random

import os
import sys

import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins

import numpy as np
import json
from collections import deque, namedtuple

import base_cont as base
# import base_pmv as base

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.uniform import Uniform
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="Pendulum-v1", help="Environment name")
parser.add_argument("-info", type=str, help="Information or name of the run")
parser.add_argument("-ep", type=str, default='3000', help="The amount of training episodes, default is 100")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-5, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float,default=0.1, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e7), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.95, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
parser.add_argument("-lmbda", type=str, default=None, required=True, help="thermal comfort linear scalarization weight")
parser.add_argument('-output', type=str, default=None, required=True, help='model directory')
args = parser.parse_args()
args.lmbda = float(args.lmbda)
args.ep = int(args.ep)

seed = args.seed
n_episodes = args.ep
GAMMA = args.gamma
TAU = args.tau
HIDDEN_SIZE = args.layer_size
BUFFER_SIZE = int(args.replay_memory)
BATCH_SIZE = args.batch_size
LR_ACTOR = args.lr
LR_CRITIC = args.lr
FIXED_ALPHA = args.alpha

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                'start_date': (6,21),
                'end_date': (8,21),
                'pmv_pickle_available': True,
                'pmv_pickle_path': 'pmv_cache.pickle',
                'lmbda': args.lmbda
                }

env = base.EnergyPlusEnv(default_args)

action_high = 1
action_low = -1
action_size = env.action_space.shape[0]
state_size = env.observation_space.shape[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):

        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Uniform(0, 1 + 1e-9)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


    def get_action(self, state, mask):
        """
        returns the action based on a squashed gaussian policy. That means the samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)

        NOTE: for action masking, action range is clamped before sampled from distribution
        """
        #state = torch.FloatTensor(state).to(device) #.unsqzeeze(0)
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Uniform(0, 1 + 1e-9)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        #print(action)

        #NOTE: action masking
        action = torch.clamp(action, mask[0], mask[1])
        return action[0]


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_size (int): Number of nodes in the network layers

        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, hidden_size, action_prior="uniform"):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR)
        self._action_prior = action_prior

        print("Using: ", device)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size).to(device)

        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=0)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


    def step(self, state, action, reward, next_state, done, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(step, experiences, GAMMA)


    def act(self, state, mask):
        """Returns actions for given state as per current policy.
        mask: tuple(low_action_bound, high_action_bound)
        """
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state, mask).detach()
        return action

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))
        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())
        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:

                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, TAU)
            self.soft_update(self.critic2, self.critic2_target, TAU)



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
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
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def test():
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed, hidden_size=HIDDEN_SIZE, action_prior='uniform')
    checkpoint = torch.load(args.output + '/model.pt')
    agent.actor_local.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_local.eval()

    steps = 0
    episode_energy_reward = 0
    episode_cost_reward = 0

    cooling_actuator_value = []
    heating_actuator_value = []
    indoor_temperature = []
    outdoor_temperature = []
    thermal_comfort = []

    for i_episode in range(1):
        state = env.reset()
        print('state', state)
        state = state.reshape((1,state_size))
        done = False

        while not done:
            mask = (-1, 1)
            action = agent.act(state, mask)
            action_v = action[0].numpy()

            action_v = np.clip(action_v * action_high, action_low, action_high)

            next_state, reward, done, truncated, info = env.step([action_v])

            next_state = next_state.reshape((1, state_size))
            state = next_state

            steps += 1

            episode_energy_reward += info['energy_reward']
            episode_cost_reward  += info['cost_reward']
            cooling_actuator_value.append(info['actuators'][0])
            heating_actuator_value.append(info['actuators'][1])
            indoor_temperature.append(info['obs_vec'][1])
            outdoor_temperature.append(info['obs_vec'][0])
            thermal_comfort.append(info['comfort_reward'])
            if done:
                break

        steps_start = 10
        steps = 310
        size = steps - steps_start

        print('##########################')
        print('EP reward:', episode_energy_reward)
        print('##########################')

        x = list(range(size))
        fig, ax1 = plt.subplots()

        acceptable_pmv = 0.7

        plt.title('acceptable_pmv: {}'.format(acceptable_pmv))
        ax1.set_xlabel('steps')
        ax1.set_ylabel('Actuators Setpoint Temperature (*C)', color='tab:blue')
        cooling_actuator = ax1.plot(x, cooling_actuator_value[steps_start:steps], label='cooling actuator value')
        heating_actuator = ax1.plot(x, heating_actuator_value[steps_start:steps], label='heating actuator value')
        indoor_temp = ax1.plot(x, indoor_temperature[steps_start:steps], label='indoor temperature')
        outdoor_temp = ax1.plot(x, outdoor_temperature[steps_start:steps], label='outdoor temperature')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # ax2 = ax1.twinx()
        # ax2.set_ylabel('PMV [-3, 3] ')
        # ax2.axhline(y=acceptable_pmv, color='black',linestyle='--')
        # ax2.axhline(y=-acceptable_pmv, color='black',linestyle='--')
        # thermal_comf = ax2.plot(x, thermal_comfort[steps_start:steps], color='black')

        interactive_legend = plugins.InteractiveLegendPlugin([
            cooling_actuator,
            heating_actuator,
            indoor_temp,
            outdoor_temp,
            #thermal_comf
        ],[
            'cooling actuator setpoint',
            'heating actuator setpoint',
            'indoor temperature',
            'outdoor temperature',
            #'thermal comfort'
        ])
        plugins.connect(fig, interactive_legend)

        html_plot = mpld3.fig_to_html(fig)

        with open(args.output + '/interactive_plot.html', 'w') as f:
            f.write(html_plot)

        ax1.legend()
        fig.tight_layout()
        plt.savefig(args.output + '/plot')
        plt.show()


test()
