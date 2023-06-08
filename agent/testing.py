
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json

import base_cont as base
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

f_name = './logs/scores-base.txt'

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def plotting_base():
    with open(f_name, 'r') as scores_file:
        scores = scores_file.read().split('\n')
        del scores[-1]
        #print(scores)

        plt.figure(figsize=(30,5))
        scores = np.array([-float(x) for x in scores])
        scores_idx = np.array(list(range(len(scores))))
        plt.plot(scores_idx, scores, 'ob-', label='Points')

        window = 10
        moving_avg = np.convolve(scores, np.ones(10) / window, mode='valid')
        plt.plot(range(window - 1, len(scores)), moving_avg, 'r-', label='Moving Average')

        #plt.yscale('log')
        plt.ylabel('Energy Consumption for Summer Period 6/21 ~ 8/21 (J)')
        plt.xlabel('Episode')
        plt.show()
        #plt.savefig('./pic/ac_base')

####################################################3
HIDDEN_SIZE=128
class Actor(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Actor, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_shape, HIDDEN_SIZE),
                                 nn.ReLU(),
                                 nn.Linear(HIDDEN_SIZE,HIDDEN_SIZE),
                                 nn.ReLU(),
                                 )
        self.mean = nn.Sequential(nn.Linear(HIDDEN_SIZE, output_shape),
                                  nn.Tanh())                    # tanh squashed output to the range of -1..1
        self.variance =nn.Sequential(nn.Linear(HIDDEN_SIZE, output_shape),
                                     nn.Softplus())             # log(1 + e^x) has the shape of a smoothed ReLU

    def forward(self,x):
        x = self.net(x)
        return self.mean(x), self.variance(x)

def sample(mean, variance):
    """
    Calculates the actions, log probs and entropy based on a normal distribution by a given mean and variance.

    ====================================================

    calculate log prob:
    log_prob = -((action - mean) ** 2) / (2 * var) - log(sigma) - log(sqrt(2 *pi))

    calculate entropy:
    entropy =  0.5 + 0.5 * log(2 *pi * sigma)
    entropy here relates directly to the unpredictability of the actions which an agent takes in a given policy.
    The greater the entropy, the more random the actions an agent takes.

    """
    sigma = torch.sqrt(variance)
    m = Normal(mean, sigma)
    actions = m.sample()
    actions = torch.clamp(actions, -1, 1) # usually clipping between -1,1 but pendulum env has action range of -2,2
    logprobs = m.log_prob(actions)
    entropy = m.entropy()  # Equation: 0.5 + 0.5 * log(2 *pi * sigma)
    #print('ENTROPY?', entropy)

    return actions, logprobs, entropy

checkpoint_path = './model/checkpoint-base.pt'

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

def test_model():
    # setting up env, actor for test run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_model = torch.load(checkpoint_path)
    episode = test_model['episode']
    env = base.EnergyPlusEnv(default_args)
    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.shape[0]

    actor = Actor(input_shape, output_shape)
    actor.load_state_dict(test_model['actor_state_dict'])
    # values to keep track
    steps = 0
    cooling_actuator_value = []
    heating_actuator_value = []
    thermal_comfort = []

    # simulation run
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        state = torch.from_numpy(state).float()
        if env.b_during_sim():
            mean, variance = actor(state.unsqueeze(0).to(device))
            action, logprob, entropy = sample(mean.cpu(), variance.cpu())
            next_state, reward, done, truncated, info = env.step(action[0].numpy())
            steps += 1
            episode_reward += reward
            state = next_state

            thermal_comfort.append(info['comfort_reward'])
            cooling_actuator_value.append(info['actuators'][0])
            heating_actuator_value.append(info['actuators'][1])
        else:
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            print('\r skipping... date', str(info['date']), end='', flush=True)
            continue

    # save test value
    f_name = './logs/model_test.json'
    with open(f_name, 'w') as test_file:
        data = {
            'steps': steps,
            'comfort_rewards': thermal_comfort,
            'cooling_actuator': cooling_actuator_value,
            'heating_actuator': heating_actuator_value
        }
        test_file.write(json.dumps(data))

    x = list(range(steps))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('steps')
    ax1.set_ylabel('Actuators Setpoint Temperature (*C)', color='tab:blue')
    ax1.plot(x, cooling_actuator_value, 'b-')
    ax1.plot(x, heating_actuator_value, 'r-')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PMV (%) ')
    ax2.plot(x, thermal_comfort, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    #test_model()
    plotting_base()
