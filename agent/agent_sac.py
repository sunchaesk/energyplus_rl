# -*- coding: utf-8 -*-
"""
BY571/SAC.py
"""

# import base_pmv as base
import base_cont as base

import sys
import numpy as np
import random

import matplotlib.pyplot as plt
import gym
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal

import torch.optim as optim
import time
from torch.utils.tensorboard import SummaryWriter
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        dist = Normal(0, 1)
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
        dist = Normal(0, 1) # loc, scale
        #print('dist', dist)
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


def save_reward(score:float) -> None:
    f_name = './logs/sac-scores.txt'
    with open(f_name, 'a') as scores_f:
        scores_f.write(str(score) + '\n')

def SAC(n_episodes=200000, max_t=500, print_every=2, load=True):

    scores_deque = deque(maxlen=100)
    average_100_scores = []

    start_episode = 0
    if load:
        try:
            checkpoint = torch.load('./model/sac-checkpoint.pt')
            agent.actor_local.load_state_dict(checkpoint['actor_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
            agent.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
            agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
            agent.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
            start_episode = checkpoint['episode']
        except:
            print('ERROR loading model... starting training from scratch')

    for i_episode in range(start_episode + 1, n_episodes+1):
    # for i_episode in range(1):

        state = env.reset()
        state = state.reshape((1,state_size))
        score = 0
        #for t in range(max_t):
        done = False

        t = 0

        # DEBUG
        rewards = []
        comfort = []
        act = []
        comfort_bound_high = []
        comfort_bound_low = []
        action_before_clip = []
        action_after_clip = []

        while not done:
            mask = env.masking_valid_actions(scale=(-1, 1))

            action = agent.act(state, mask)
            action_pre_clip = action.numpy()
            action_v = np.clip(action_pre_clip*action_high, action_low, action_high)
            #print(action_v)
            next_state, reward, done, truncated, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))


            agent.step(state, action, reward, next_state, done, t)
            t += 1

            # DEBUG
            action_before_clip.append(action_pre_clip[0])
            rewards.append(reward)
            comfort.append(info['comfort_reward'])
                # temp = env.masking_valid_actions()
                # comfort_bound_high.append(temp[1])
                # comfort_bound_low.append(temp[0])

            state = next_state

            #DEBUG
            score += info['energy_reward']
            act.append(action_v[0])
            rewards.append(reward)
            comfort.append(info['comfort_reward'])


            if done:
                break

        # DEBUG
        env.pickle_save_pmv_cache()
        start = 0
        end = 300
        x = list(range(end - start))

        print(rewards)
        print('mean', np.mean(rewards))
        # print(x)
        # print(act)

        fig, ax1 = plt.subplots()
        ax1.set_title('First 300 steps of training episode {}'.format(i_episode))
        ax1.scatter(x[start:end], act[start:end], color='red')

        ax2 = ax1.twinx()
        ax2.plot(x[start:end], comfort[start:end], color='blue', linestyle='-')
        ax2.axhline(0.7, color='black', linestyle='--')
        ax2.axhline(-0.7, color='black', linestyle='--')
        fig.tight_layout()
        plt.show()
        time.sleep(1)
        plt.close('all')

        scores_deque.append(score)
        # writer.add_scalar("Reward", score, i_episode)
        # writer.add_scalar("average_X", np.mean(scores_deque), i_episode)
        average_100_scores.append(np.mean(scores_deque))

        print('##################################')
        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        print('##################################')
        save_reward(score)
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))
            print('\rSaving Current model at episode {}'.format(i_episode))
            torch.save({
                'episode': i_episode,
                'actor_state_dict': agent.actor_local.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic1_state_dict': agent.critic1.state_dict(),
                'critic2_state_dict': agent.critic2.state_dict(),
                'critic1_optimizer': agent.critic1_optimizer.state_dict(),
                'critic2_optimizer': agent.critic2_optimizer.state_dict()
            }, './model/sac-checkpoint.pt')


    # for loop end
    #torch.save(agent.actor_local.state_dict(), 'pendulum' + ".pt")






parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="Pendulum-v1", help="Environment name")
parser.add_argument("-info", type=str, help="Information or name of the run")
parser.add_argument("-ep", type=int, default=100, help="The amount of training episodes, default is 100")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-5, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float,default=0.1, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=2, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
args = parser.parse_args()


default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,
                'start_date': (6,21),
                'end_date': (8,21),
                'pmv_pickle_available': True,
                'pmv_pickle_path': './pmv_cache.pickle'
                }

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed
    n_episodes = args.ep
    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size        # minibatch size
    LR_ACTOR = args.lr         # learning rate of the actor
    LR_CRITIC = args.lr        # learning rate of the critic
    FIXED_ALPHA = args.alpha
    FIXED_ALPHA = 2500
    print('################3')
    print("ALPHA", FIXED_ALPHA)
    print('################3')
    #saved_model = args.saved_model
    saved_model = None

    t0 = time.time()
    #writer = SummaryWriter("temp/" + 'pendulum')
    #env = gym.make(env_name)
    env = base.EnergyPlusEnv(default_args)

    # action_high = env.action_space.high[0] #NOTE: action [-1, 1] converted to range anywas in the base py
    # action_low = env.action_space.low[0]
    action_high = 1
    action_low = -1

    torch.manual_seed(seed)
    np.random.seed(seed)

    state_size = env.observation_space.shape[0]
    print('###########')
    print(state_size)
    print('###########')
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,hidden_size=HIDDEN_SIZE, action_prior="uniform") #"normal"

    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model))
        play()
    else:
        SAC(n_episodes=110, max_t=100000, print_every=args.print_every,load=True)
    t1 = time.time()
    env.close()
    print("training took {} min!".format((t1-t0)/60))
