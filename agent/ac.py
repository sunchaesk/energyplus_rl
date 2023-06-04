import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v1')
env.reset(seed=args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)

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

        self.model = Policy()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        m = Categorical(probs)

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

    def train(max_episodes=500):
        running_reward = 10
        step_cnt = 0
        for ep in range(max_episodes):
            done = False
            episode_reward = 0
            observation = self.env.reset()
            while not done:
                # get action
                action = self.select_action(state)
                # take the action
                state, reward, done, _ = env.step(action)

                self.model.rewards.append(reward)
                episode_reward += reward
                if done:
                    break

            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
            self.finish_episode()

            if ep % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    ep, episode_reward, running_reward))

        # check if we have "solved" the cart pole problem
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break






def main():
    env = gym.make('CartPole-v1')
    env.reset()
    running_reward = 10
    ac = ActorCritic(env, env.observation_space.shape[0], env.action_space.n)

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 10000):

            # select action from policy
            action = ac.select_action(state)

            # take the action
            state, reward, done,  _ = env.step(action)

            if args.render:
                env.render()

            ac.model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        ac.finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
