
import base as base
import sys
import matplotlib.pyplot as plt

import gym
import torch
import numpy as np
from PPO import device, PPO_discrete
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse



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
parser.add_argument('--T_horizon', type=int, default=2048, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e25, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.95, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=200, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0.01, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
opt = parser.parse_args()
print(opt)

default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

def graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, episode):
    start = 10
    end = 310
    x = list(range(end - start))

    print(len(x))
    print(len(cooling_setpoints[start:end]))

    fig, ax1 = plt.subplots()
    ax1.set_title('10 - 310 steps of training {} episodes'.format(episode))
    ax1.scatter(x, cooling_setpoints[start:end], color='red')
    ax1.plot(x, outdoor_temperatures[start:end], linestyle='--', color='green')
    ax1.plot(x, indoor_temperatures[start:end], linestyle='--', color='magenta')

    ax2 = ax1.twinx()
    ax2.plot(x, cost_signals[start:end])
    fig.tight_layout()
    plt.savefig('./logs/curr.png')

def main():
    #env = gym.make(EnvName[EnvIdex])
    env = base.EnergyPlusEnv(default_args)
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

    if not os.path.exists('model'): os.mkdir('model')
    model = PPO_discrete(**kwargs)
    Loadmodel = True
    if Loadmodel: model.load()

    scores = []
    episodes = 0
    cost_signals = []
    cooling_setpoints = []
    outdoor_temperatures = []
    indoor_temperatures = []


    traj_lenth = 0
    total_steps = 0
    while total_steps < Max_train_steps:
        s, done, steps, ep_r = env.reset(), False, 0, 0

        '''Interact & trian'''
        while not done:
            #print(total_steps)
            traj_lenth += 1
            steps += 1

            a, pi_a = model.select_action(torch.from_numpy(s).float().to(device))  #stochastic policy
            # a, pi_a = model.evaluate(torch.from_numpy(s).float().to(device))  #deterministic policy

            s_prime, r, done, truncated, info = env.step(a)

            cost_signals.append(info['cost_signal'])
            cooling_setpoints.append(info['cooling_actuator_value'])
            outdoor_temperatures.append(s_prime[0])
            indoor_temperatures.append(s_prime[1])

            dw = False
            model.put_data((s, a, r, s_prime, pi_a, done, dw))
            s = s_prime
            ep_r += r

            if traj_lenth % T_horizon == 0:
                a_loss, c_loss, entropy = model.train()
                traj_lenth = 0


            total_steps += 1

            if done:
                '''save model'''
                if episodes != 0 and episodes % 2 == 0:
                    model.save(total_steps)
                graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, episodes)
                scores.append(ep_r)
                episodes += 1
                f_name = './logs/scores.txt'
                with open(f_name, 'a') as scores_f:
                    scores_f.write(str(ep_r) + '\n')

                cooling_setpoints = []
                cost_signals = []
                indoor_temperatures = []
                outdoor_temperatures = []

                ep_r = 0

if __name__ == '__main__':
    main()