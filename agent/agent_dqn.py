
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

'''
0 -TH ACTION VALUE: -80330270668.17052
1 -TH ACTION VALUE: -79741665082.9226
2 -TH ACTION VALUE: -79194870611.75945
3 -TH ACTION VALUE: -78658119926.9327
4 -TH ACTION VALUE: -78123605697.18463
5 -TH ACTION VALUE: -77603097344.10558
6 -TH ACTION VALUE: -77088116657.93875
7 -TH ACTION VALUE: -76575868086.57722
8 -TH ACTION VALUE: -76070032715.13345
9 -TH ACTION VALUE: -75559609303.27484
10 -TH ACTION VALUE: -75052133972.88834
11 -TH ACTION VALUE: -74568785819.23244
12 -TH ACTION VALUE: -74070011217.51329
13 -TH ACTION VALUE: -73584745374.28305
14 -TH ACTION VALUE: -73093079619.46715
15 -TH ACTION VALUE: -72617408338.65147
16 -TH ACTION VALUE: -72144055355.22818
17 -TH ACTION VALUE: -71659262236.06543
18 -TH ACTION VALUE: -71181891828.90132
19 -TH ACTION VALUE: -70713618641.46324
'''

# NOTE: tensorboard stuff
logdir = os.path.join(
    './logs/', 'agent_dqn',
    datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)
# # tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

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
        tf.profiler.experimental.start('logs')
        step_cnt = 0
        with writer.as_default(): # Tensorboard logging
            for ep in range(max_episodes):
                done, episode_reward = False, 0
                observation = self.env.reset()
                while not done:
                    action = self.model.get_action(observation)
                    # next_observation, reward, done, _ = self.env.step(action)
                    next_observation, reward, done, _, _ = self.env.step(action)
                    print('REWARD', reward, 'STEP:', step_cnt, 'EPS:', self.model.epsilon)
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
                        tf.profiler.experimental.stop()
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

def testing_static_only(action_space_n):
    action_n_scores = []
    for i in range(action_space_n):
        action_n_scores.append(test_model_static_action(i))

    for i in range(len(action_n_scores)):
        print(i, '-TH ACTION VALUE:', action_n_scores[i])

    return action_n_scores

def generate_graph():

    with open('saved_scores.csv','r') as scores:
        scores_list = scores.read()
        scores_list.replace('\n', '').replace('-','')
        scores_list = scores_list.split(',')
        scores_list = [float(x) for x in scores_list]
    np_scores_array = np.asarray(scores_list)
    np_names_array = np.array(['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','model'])

    plt.bar(np_names_array, np_scores_array)
    plt.yscale('log')
    plt.ylabel('Energy Consumption (J)')
    plt.xlabel('Thermostat control')
    plt.show()
    # scores_list = list(np_scores_array)

def main():
    env = base.EnergyPlusEnv(default_args)
    agent = Agent(env)
    agent.train(max_episodes=500)


if __name__ == "__main__":
    # NOTE: generating static energy consumption values
    # env = base.EnergyPlusEnv(default_args)
    # scores = testing_static_only(env.action_space.n)
    # output_string = ','.join(scores)
    # with open('static_scores.csv', 'w+') as output_file:
    #     output_file.write(output_string)

   # main()
   # env = base.EnergyPlusEnv(default_args)
   # testing_full(env.action_space.n)
   # score = test_load_model()
   generate_graph()

    # NOTE: Profiling Stuff
    # cProfile.run('main()', 'restats')
    # p = pstats.Stats('restats')
    #print(type(p))
    #p.strip_dirs().sort_stats(-1).print_stats()
    # p.strip_dirs().sort_stats('tottime').print_stats().dump_stats('./restats.txt')
    #main()
    # env = gym.make('CartPole-v1')
