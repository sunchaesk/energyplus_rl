import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

#import gymnasium as gym # DEPRECATED: causes error (not sure y)
import gym
import argparse
import numpy as np
import matplotlib.pyplot as plt

import base

import os
import random
from collections import deque

tf.keras.backend.set_floatx('float64')

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

#args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = 128

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.gamma = 0.95


        self.model = self.create_model()

    def create_model(self):
        # model = tf.keras.Sequential([
        #     Input((self.state_dim,)),
        #     Dense(32, activation='relu'),
        #     Dense(16, activation='relu'),
        #     Dense(self.action_dim)
        # ])
        model = tf.keras.Sequential([
            Dense(48, input_dim=self.state_dim, activation='relu'),
            Dense(48, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(loss='mse',
                      optimizer=Adam(self.learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.model = DQNAgent(self.state_dim, self.action_dim)
        self.target_model = DQNAgent(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(40):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.buffer.batch_size), actions] = rewards + (1-done) * next_q_values * self.model.gamma
            self.model.train(states, targets)

    def load(self, name):
        self.model.model.load_weights(name)

    def save(self, name):
        self.model.model.save_weights(name)

    def train(self, max_episodes=1000, check_point_num=20):
        scores = np.array([])
        start_episode = 0

        try:
            scores = np.genfromtxt('saved_scores.csv', delimiter=',')
            start_episode = len(scores)
            self.load('./model/agent-{}'.format(start_episode))
        except:
            print('## saved_scores.csv empty. Skipping prev scores loading')

        for ep in range(start_episode, max_episodes + 1):
            if ep % check_point_num == 0 and ep != 0:
                self.save('./model/agent-{}'.format(ep))
                np.savetxt('saved_scores.csv', scores, delimiter=',')

            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                print('ACTION:', action)
                print('EPISODE', ep)
                next_state, reward, done, _, _ = self.env.step(action)
                print('NEXT_STATE', next_state)
                print('REWARD', reward)
                self.buffer.put(state, action, reward, next_state, done) # env returns negative rewards
                total_reward += reward * -1 # add positive values to total_reward
                state = next_state

            if self.buffer.size() >= self.buffer.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            scores = np.append(scores, [total_reward])

        plt.plot(scores)
        plt.ylabel('energy consumption')
        plt.xlabel('episodes')
        plt.title('E+ Reinforcement Learning')
        plt.show()

        # for ep in range(max_episodes):
        #     done, total_reward = False, 0
        #     state = self.env.reset()
        #     while not done:
        #         action = self.model.get_action(state)
        #         next_state, reward, done, _ = self.env.step(action)
        #         self.buffer.put(state, action, reward*0.01, next_state, done)
        #         total_reward += reward
        #         state = next_state
        #     if self.buffer.size() >= self.buffer.batch_size:
        #         self.replay()
        #     self.target_update()
        #     print('EP{} EpisodeReward={}'.format(ep, total_reward))


# def main():
#     env = gym.make('CartPole-v1')
#     agent = Agent(env)
#     agent.train(max_episodes=1000)

default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2
                }

def main():
    env = base.EnergyPlusEnv(default_args)
    agent = Agent(env)
    agent.train(max_episodes=500, check_point_num=2)

if __name__ == "__main__":
    main()
