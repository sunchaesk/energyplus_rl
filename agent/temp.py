
import base_cont as base

import gym
import numpy as np
from collections import namedtuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# tensorflow/logging setup stuff/
writer = SummaryWriter()
f_name = './logs/scores-'
f_name += datetime.now().strftime('%m-%d-%H:%M')
f_name += '.txt'

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, env):
        super(Policy, self).__init__()
        self.env = env
        self.obs_space = env.observation_space.shape[0]
        self.affine1 = nn.Linear(self.obs_space, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 2)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


class ActorCritic():
    def __init__(self, env, state_dim, action_dim, learning_rate=0.005):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.named_tup = namedtuple('SavedAction', ['log_prob', 'value'])
        self.learning_rate = learning_rate
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

        self.save_freq = 10 # num of episodes for saving

        self.model = Policy(env)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        m = Normal(probs)

        action = m.sample()

        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()

    def finish_episode(self):
        '''
        Calculates actor and critic loss and performs backprop
        '''
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            policy_losses.append(-log_prob * advantage)

            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        self.optimizer.zero_grad()

        # sum all values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # performs backprop
        loss.backward()
        self.optimizer.step()

        # reset
        del self.model.rewards[:]
        del self.model.saved_actions[:]
        # return loss for tensorboard logging
        return loss

    def train(self,max_episodes=500):
        step_cnt = 0
        for ep in range(max_episodes):
            if ep % self.save_freq == 0 and ep != 0:
                torch.save(self.model.state_dict(), './model/save-' +str(ep) + '.pt')
            done = False
            episode_reward = 0
            observation = self.env.reset()
            reward = 0
            info = {}
            while not done:
                # get action
                if self.env.b_during_sim():
                    # if during sim NN to select action
                    action = self.select_action(observation)
                else:
                    # if not during sim, random sample, important that NN is not triggered
                    action = self.env.action_space.sample()[0]
                # take the action
                observation, reward, done, truncated, info = self.env.step(action)
                print('REWARD', reward, 'ACTION:', action)
                if reward == 0:
                    #print('Cont... Date:', info.get('date', None))
                    continue
                #
                self.model.rewards.append(reward)
                episode_reward += reward
                if done:
                    loss = self.finish_episode()
                    writer.add_scalar('Loss/Train', loss, ep)
                    writer.add_scalar('EP-Reward/Train', episode_reward, ep)
                    print('Episode {}\tLast reward: {:.2f}\t'.format(
                        ep, episode_reward)
                    )
                    save_reward(episode_reward)
                    print('Simulation Ended... Starting new simulation')
                    break

def save_reward(score:float) -> None:
  with open(f_name, 'a') as scores_f:
    scores_f.write(str(score) + '\n')

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': True,# for some reasons if not annual, funky results
                'start_date': (6,21),
                'end_date': (8,21)
                }

def main():
    env = base.EnergyPlusEnv(default_args)
    agent = ActorCritic(env,
                        env.observation_space.shape[0],
                        env.action_space)
    agent.train(500)


if __name__ == "__main__":
    main()
