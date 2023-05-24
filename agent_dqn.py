

import base
from datetime import datetime
import os
import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

tf.keras.backend.set_floatx("float64")

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    def store(self, state, action, reward, next_state, done):
        # S, A, R, A', done
        self.buffer.append([state, action, reward, next_state, done])


# TODO: have to give in -x flag
default_args = {'idf': '/home/ck/Downloads/Files/in.idf',
                'epw': '/home/ck/Downloads/Files/weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2
}

if __name__ == "__main__":
    print('main')
    env = base.EnergyPlusEnv(default_args)
    print('action_space:', end='')
    print(env.action_space)
    scores = []
    for episode in range(2):
        state = env.reset()
        done = False
        score = 0

        while not done:
            #env.render()
            # action = env.action_space.sample()
            #action = 22.0
            ret = n_state, reward, done, info, STUFF = env.step(0)
            #print('RET STUFF:', ret)
            score+=reward
            # print('DONE?:', done)
            print('Episode:{} Reward:{} Score:{}'.format(episode, reward, score))

        scores.append(score)
    print("SCORES: ", scores)
    print("TRULY DONE?") # YES, but program doesn't terminate due to threading stuff?
