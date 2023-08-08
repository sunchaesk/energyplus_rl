import time
import random

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import deque, namedtuple

import base as base
import base2 as base2
# import base_pmv as base

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.distributions.uniform import Uniform
import torch.optim as optim
import argparse

from PPO import device, PPO_discrete
from PPO_CAPS import PPO_discrete_CAPS

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=True, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--T_horizon', type=int, default=1253, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e25, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.99, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=50, help='PPO update times')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=500, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0.0005, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.9995, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
print(opt)


def test_caps(checkpoint_path, graph=True):
    env = base2.EnergyPlusEnv(default_args)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('Env: Eplus','  state_dim:',state_dim,'  action_dim:',action_dim,'   Random Seed:',seed)
    print('\n')

def test(checkpoint_path, graph=True):
    #env = gym.make(EnvName[EnvIdex])
    env = base2.EnergyPlusEnv(default_args)
    #eval_env = gym.make(EnvName[EnvIdex])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    T_horizon = opt.T_horizon
    render = opt.render
    Loadmodel = opt.Loadmodel
    ModelIdex = opt.ModelIdex #which model to load
    Max_train_steps = opt.Max_train_steps #in steps
    eval_interval = opt.eval_interval #in steps
    save_interval = opt.save_interval #in steps

    seed = opt.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('Env: Eplus','  state_dim:',state_dim,'  action_dim:',action_dim,'   Random Seed:',seed)
    print('\n')


    kwargs = {
        "env_with_Dead": True,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "lambd": opt.lambd,
        "net_width": opt.net_width,
        "lr": opt.lr,
        "clip_rate": opt.clip_rate,
        "K_epochs": opt.K_epochs,
        "batch_size": opt.batch_size,
        "l2_reg":opt.l2_reg,
        "entropy_coef":opt.entropy_coef,  #hard env needs large value
        "adv_normalization":opt.adv_normalization,
        "entropy_coef_decay": opt.entropy_coef_decay,
    }

    model = PPO_discrete(**kwargs)
    episode = model.load(checkpoint_path)


    print('EPISODE:', episode)
    time.sleep(3)

    steps = 0
    episode_reward = 0
    cooling_setpoint = []
    cost_signal = []
    indoor_temp = []
    outdoor_temp = []

    thermal_comforts = []

    cost_reward_sum = 0

    for i_episode in range(1):
        state = env.reset()
        print('state', state)

        while True:
            a, pi_a = model.select_action(torch.from_numpy(state).float().to(device))

            next_state, reward, done, truncated, info = env.step(a)

            state = next_state


            #print(state[0][0])
            steps += 1
            episode_reward += info['cost_reward']
            indoor_temp.append(next_state[1])
            outdoor_temp.append(next_state[0])
            cooling_setpoint.append(info['cooling_actuator_value'])
            cost_signal.append(info['cost_signal'])
            thermal_comforts.append(info['comfort_reward'])
            # episode_reward += info['energy_reward']
            # cooling_actuator_value.append(info['actuators'][0])
            # heating_actuator_value.append(info['actuators'][1])
            # indoor_temperature.append(state[0][1])
            # outdoor_temperature.append(state[0][0])
            # thermal_comfort.append(info['comfort_reward'])
            if done:
                break


    # calculate total variance
    total_variance = 0
    for i in range(1, len(cooling_setpoint)):
        total_variance += abs(cooling_setpoint[i] - cooling_setpoint[i - 1])
    print('TOTAL VARIANCE', total_variance)
    # INFO w/out CAPS
    #

    steps_start = 110
    steps = 1110
    size = steps - steps_start
    print('##########################')
    print('EP reward:', episode_reward)
    print('##########################')

    print(episode_reward)

    x = list(range(size))
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('steps')
    ax1.set_ylabel('Actuators Setpoint Temperature (*C)', color='tab:blue')
    ax1.plot(x, cooling_setpoint[steps_start:steps], 'b-', label='cooling actuator value')
    ax1.plot(x, indoor_temp[steps_start:steps], 'g--', label='indoor temperature')
    ax1.plot(x, outdoor_temp[steps_start:steps], 'm--', label='outdoor temperature')
    # ax1.plot(x, indoor_temperature[steps_start:steps], 'g-', label='indoor temperature')
    # ax1.plot(x, outdoor_temperature[steps_start:steps], 'c-', label='outdoor temperature')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('PMV [-3, 3] ')
    ax2.plot(x, cost_signal[steps_start:steps], color='black', label='cost signal')
    # ax2.tick_params(axis='y', labelcolor='black')

    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    if graph:
        plt.show()

    # computing thermal comfort values
    avg_thermal_comfort = sum(thermal_comforts) / (len(thermal_comforts) + 1)
    return episode_reward, cooling_setpoint, indoor_temp, outdoor_temp, cost_signal, total_variance, avg_thermal_comfort


def test_max():
    env = base.EnergyPlusEnv(default_args)

    steps = 0
    episode_reward = 0
    cooling_setpoint = []
    cost_signal = []
    indoor_temp = []
    outdoor_temp = []

    cost_reward_sum = 0

    for i_episode in range(1):
        state = env.reset()

        while True:
            a = env.action_space.n - 1

            next_state, reward, done, truncated, info = env.step(a)

            state = next_state

            #print(state[0][0])
            steps += 1
            episode_reward += info['cost_reward']
            print(info['cost_reward'])
            indoor_temp.append(next_state[1])
            outdoor_temp.append(next_state[0])
            cooling_setpoint.append(info['cooling_actuator_value'])
            cost_signal.append(info['cost_signal'])
            # episode_reward += info['energy_reward']
            # cooling_actuator_value.append(info['actuators'][0])
            # heating_actuator_value.append(info['actuators'][1])
            # indoor_temperature.append(state[0][1])
            # outdoor_temperature.append(state[0][0])
            # thermal_comfort.append(info['comfort_reward'])
            if done:
                break


    steps_start = 10
    steps = 710
    size = steps - steps_start
    print('##########################')
    print('EP reward:', episode_reward)
    print('##########################')

    print(episode_reward)

    # x = list(range(size))
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('steps')
    # ax1.set_ylabel('Actuators Setpoint Temperature (*C)', color='tab:blue')
    # ax1.plot(x, cooling_setpoint[steps_start:steps], 'b-', label='cooling actuator value')
    # ax1.plot(x, indoor_temp[steps_start:steps], 'g--', label='indoor temperature')
    # ax1.plot(x, outdoor_temp[steps_start:steps], 'm--', label='outdoor temperature')
    # # ax1.plot(x, indoor_temperature[steps_start:steps], 'g-', label='indoor temperature')
    # # ax1.plot(x, outdoor_temperature[steps_start:steps], 'c-', label='outdoor temperature')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('PMV [-3, 3] ')
    # ax2.plot(x, cost_signal[steps_start:steps], color='black', label='cost signal')
    # # ax2.tick_params(axis='y', labelcolor='black')

    # ax1.legend()
    # ax2.legend()
    # fig.tight_layout()
    #plt.show()
    return episode_reward
# checkpoint_path = './model/test-sac-checkpoint.pt'
if __name__ == "__main__":

    for i in range(40):
        caps = test('checkpoint2')

    for i in range(40):
        mmm = test_max()

    # no_caps = test('checkpoint')
    # caps = test('checkpoint2')

    # cooling_setpoint_no_caps = no_caps[1]
    # cooling_setpoint_caps = caps[1]
    # indoor_temp = no_caps[2]
    # outdoor_temp = no_caps[3]
    # cost_signals = no_caps[4]

    # steps_start = 110
    # steps = 1110
    # size = steps - steps_start
    # # print('##########################')
    # # print('EP reward:', episode_reward)
    # # print('##########################')

    # #print(episode_reward)

    # x = list(range(size))
    # fig, ax1 = plt.subplots()
    # ax1.set_xlabel('steps')
    # ax1.set_ylabel('Actuators Setpoint Temperature (*C)', color='tab:blue')
    # ax1.plot(x, cooling_setpoint_caps[steps_start:steps], 'b-', label='Actuator CAPS')
    # ax1.plot(x, cooling_setpoint_no_caps[steps_start:steps], 'r-', label='Actuator No CAPS')
    # #ax1.plot(x, indoor_temp[steps_start:steps], 'g--', label='indoor temperature')
    # ax1.plot(x, outdoor_temp[steps_start:steps], 'm--', label='outdoor temperature')
    # # ax1.plot(x, indoor_temperature[steps_start:steps], 'g-', label='indoor temperature')
    # # ax1.plot(x, outdoor_temperature[steps_start:steps], 'c-', label='outdoor temperature')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()
    # ax2.set_ylabel('PMV [-3, 3] ')
    # ax2.plot(x, cost_signals[steps_start:steps], color='black', label='cost signal')
    # # ax2.tick_params(axis='y', labelcolor='black')

    # ax1.legend()
    # ax2.legend()
    # fig.tight_layout()
    # plt.show()
