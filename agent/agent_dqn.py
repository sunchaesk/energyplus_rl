
import base

import argparse
from datetime import datetime
import os
import random
from collections import deque

# debugging stuff
import io
import cProfile
import pstats
import sys
from pympler import muppy, summary
#installed py-spy, pympler

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

logdir = os.path.join(
    './logs/', 'agent_dqn',
    datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

class ReplayBuffer:
    def __init__(self, capacity=1200, batch_size=256):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def store(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)

class DQN:
    def __init__(self, state_dim, action_dim, epsilon=1.0, learning_rate=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.learning_rate=learning_rate

        self.model = self.nn_model()

    def nn_model(self):
        model = tf.keras.Sequential(
            [
                Input((self.state_dim,)),
                # Dense(12, activation='relu'),
                Dense(12, activation='relu'),
                Dense(self.action_dim),
            ]
        )
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1)

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.model = DQN(self.state_dim, self.action_dim)
        self.target_model = DQN(self.state_dim, self.action_dim)
        self.update_target()
        self.buffer = ReplayBuffer()

    def update_target(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay_experience(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(self.buffer.batch_size), actions] = (
                rewards + (1 - done) * next_q_values * self.model.gamma
            )
            self.model.train(states, targets)

    def train(self, max_episodes=500, save_fname='save'):
        step_cnt = 0
        with writer.as_default(): # Tensorboard logging
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                observation = self.env.reset()
                while not done:
                    action = self.model.get_action(observation)
                    # next_observation, reward, done, _ = self.env.step(action)
                    next_observation, reward, done, _, _ = self.env.step(action)
                    print('STEP:', step_cnt, 'EPS:', self.model.epsilon)
                    # print(type(next_observation))
                    if not isinstance(next_observation, np.ndarray):
                        done = False
                        continue

                    # if step_cnt == 12500:
                    #     all_objs = muppy.get_objects()
                    #     summ = summary.summarize(all_objs)
                    #     summary.print_(summ)
                    #     return

                    #NOTE: save models after 12500 steps
                    step_cnt += 1
                    if step_cnt == 12500:
                        self.model.model.save('./model/' + save_fname)
                        print('############\n\n')
                        print('MODEL SAVED AS: ./model/save.h5\n\n')
                        print('############')
                        return


                    self.buffer.store(
                        observation, action, reward, next_observation, done
                    )
                    episode_reward += reward
                    observation = next_observation
                if self.buffer.size() >= self.buffer.batch_size:
                    self.replay_experience()
                self.update_target()
                print(f"Episode#{ep} Reward:{episode_reward}")
                tf.summary.scalar("episode_reward", episode_reward, step=ep)
                writer.flush()


default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2
                }

def test_load_model(save_fname='save') -> int:
    agent_test = tf.keras.models.load_model('./model/' + save_fname)
    agent_test.summary()
    env = base.EnergyPlusEnv(default_args)

    done = False
    score = 0
    observation = env.reset()
    while not done:
        # agent_test.
        # action = 2 # get model to make the action

        observation = np.reshape(observation, [1, env.observation_space.shape[0]])
        q_value = agent_test.predict(observation)[0]
        action = np.argmax(q_value)

        ret = observation, reward, done, truncated, info = env.step(action)
        score += reward

    print('----------------SUMMARY-----------------')
    print('TIME:', datetime.now())
    print('MODEL:', save_fname)
    print('SCORE:', score)
    print('------------SUMMARY END-----------------')
    # return (save_fname, score)
    return score

def test_model_static_action(action_n) -> int:
    ''''
    @param: action_n from [0,action_space.n)
    '''
    env = base.EnergyPlusEnv(default_args)

    done = False
    score = 0
    observation = env.reset()
    while not done:
        ret = observation, reward, done, truncated, info = env.step(action_n)
        score += reward

    print('----------------SUMMARY-----------------')
    print('TIME:', datetime.now())
    print('ACTION_N:', action_n)
    print('SCORE:', score)
    print('------------SUMMARY END-----------------')
    return score


def testing_full(action_space_n):
    '''
    TODO: add graphs matplotlib.pyplot
    '''
    model_score = test_load_model()
    action_n_scores = []
    for i in range(action_space_n):
        action_n_scores[i] = test_model_static_action(i)

    print('MODEL_SCORE:', model_score)
    for i in range(len(action_n_scores)):
        print(i, '-TH ACTION VALUE:', action_n_scores[i])

    action_n_scores.append(model_score)
    return action_n_scores

def main():
    env = base.EnergyPlusEnv(default_args)
    agent = Agent(env)
    agent.train(max_episodes=500)


if __name__ == "__main__":
   # main()
   env = base.EnergyPlusEnv(default_args)
   testing_full(env.action_space.n)
   # test_load_model()

    # NOTE: Profiling Stuff
    # cProfile.run('main()', 'restats')
    # p = pstats.Stats('restats')
    #print(type(p))
    #p.strip_dirs().sort_stats(-1).print_stats()
    # p.strip_dirs().sort_stats('tottime').print_stats().dump_stats('./restats.txt')
    #main()
    # env = gym.make('CartPole-v1')
