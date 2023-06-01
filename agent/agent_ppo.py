
# use when testing on E+ env
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

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

logdir = os.path.join(
 'logs', 'agent_ppo',
 datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.nn_model()
        self.optimizer = Adam(0.0005) #0.0005 is actor learning rate

    def nn_model():
        state_input = Input((self.state_dim,))
        dense_1 = Dense(32, activation='relu')(state_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x : x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action():
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        action =


'''
NOTE: hyperparameters
update_freq = 5
epochs = 3
actor-lr = 0.0005
critic-lr = 0.001
clip-ratio = 0.1
GAE-lambda = 0.95
gamma = 0.99
logdir = logs
'''
if __name__ == "__main__":
    env = gym.make('Pendulum-v1')
