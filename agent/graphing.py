
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


def plot_hidden():
    data_fname1 = './logs/scores-base-128.txt'
    data_fname2 = './logs/scores-base-256.txt'
    with open(data_fname1, 'r') as data1f, open(data_fname2, 'r') as data2f:
        scores1 = data1f.read().split('\n')
        scores2 = data2f.read().split('\n')
        del scores1[-1]
        del scores2[-1]
        scores1 = np.array([-float(x) for x in scores1])
        scores2 = np.array([-float(x) for x in scores2])
        scores_idx1 = np.array(list(range(len(scores1))))
        scores_idx2 = np.array(list(range(len(scores2))))
        #print(scores)

        plt.figure(figsize=(30,5))
        # plt.plot(scores_idx1, scores1, 'b-', label='128 hidden')
        # plt.plot(scores_idx2, scores2, 'r-', label='256 hidden')

        window = 5
        label1 = f'128 nodes {window} ep moving average'
        label2 = f'256 nodes {window} ep moving average'
        moving_avg1 = np.convolve(scores1, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(scores1)), moving_avg1, 'r-', label=label1)
        moving_avg2 = np.convolve(scores2, np.ones(window) / window, mode='valid')
        plt.plot(range(window - 1, len(scores2)), moving_avg2, 'b-', label=label2)

        #plt.yscale('log')
        plt.title(f'Hidden Nodes per Layer 128 vs 256 ({window} ep Moving Average)')
        plt.ylabel('Energy Consumption for Summer Period 6/21 ~ 8/21 (J)')
        plt.xlabel('Episode')
        plt.legend()
        plt.show()
        #plt.savefig('./pic/ac_base')

def plot_horizon():
    data_fname1 = './logs/scores-base-128.txt'
    data_fname2 = './logs/scores-finite6-128.txt'
    data_fname3 = './logs/scores-finite12-128.txt'
    data_fname4 = './logs/scores-base-0.99-128.txt'
    with open(data_fname1, 'r') as data1, open(data_fname2, 'r') as data2, open(data_fname3, 'r') as data3, open(data_fname4, 'r') as data4:
        plt.figure(figsize=(30,5))
        scores1 = data1.read().split('\n')
        scores2 = data2.read().split('\n')
        scores3 = data3.read().split('\n')
        scores4 = data4.read().split('\n')
        del scores1[-1]
        del scores2[-1]
        del scores3[-1]
        del scores4[-1]
        scores1 = np.array([-float(x) for x in scores1])
        scores2 = np.array([-float(x) for x in scores2])
        scores3 = np.array([-float(x) for x in scores3])
        scores4 = np.array([-float(x) for x in scores4])
        scores1_idx = np.array(list(range(len(scores1))))
        scores2_idx = np.array(list(range(len(scores2))))
        scores3_idx = np.array(list(range(len(scores3))))
        scores4_idx = np.array(list(range(len(scores4))))

        # plt.plot(scores1_idx, scores1, 'b-', label='gamma=0.9 discounted infinite horizon')
        # plt.plot(scores2_idx, scores2, 'r-', label='undiscounted 6hr finite horizon')
        # plt.plot(scores3_idx, scores3, 'g-', label='undiscounted 12hr finite horizon')
        # plt.plot(scores4_idx, scores4, 'c-', label='gamma=0.99 discounted infinite horizon')

        # plt.plot(scores1_idx, scores1[:len(scores1)], 'b-', label='gamma=0.9 discounted infinite horizon')
        # plt.plot(scores1_idx, scores2[:len(scores1)], 'r-', label='undiscounted 6hr finite horizon')
        # plt.plot(scores1_idx, scores3[:len(scores1)], 'g-', label='undiscounted 12hr finite horizon')
        # plt.plot(scores1_idx, scores4[:len(scores1)], 'c-', label='gamma=0.99 discounted infinite horizon')

        # moving average
        window = 5
        label1 = 'gamma=0.9 discounted infinite horizon'
        label2 = 'undiscounted 6hr finite horizon'
        label3 = 'undiscounted 12hr finite horizon'
        label4 = 'gamma=0.99 discounted infinite horizon'
        moving_avg1 = np.convolve(scores1, np.ones(window)/window, mode='valid')
        moving_avg2 = np.convolve(scores2, np.ones(window)/window, mode='valid')
        moving_avg3 = np.convolve(scores3, np.ones(window)/window, mode='valid')
        moving_avg4 = np.convolve(scores4, np.ones(window)/window, mode='valid')
        # plt.plot(range(window - 1, len(scores1)), moving_avg1, 'b-', label=label1)
        # plt.plot(range(window - 1, len(scores2)), moving_avg2, 'r-', label=label2)
        # plt.plot(range(window - 1, len(scores3)), moving_avg3, 'g-', label=label3)
        # plt.plot(range(window - 1, len(scores4)), moving_avg4, 'c-', label=label4)

        plt.plot(range(window - 1, len(scores1)), moving_avg1[:len(moving_avg1)], 'b-', label=label1)
        plt.plot(range(window - 1, len(scores1)), moving_avg2[:len(moving_avg1)], 'r-', label=label2)
        plt.plot(range(window - 1, len(scores1)), moving_avg3[:len(moving_avg1)], 'g-', label=label3)
        plt.plot(range(window - 1, len(scores1)), moving_avg4[:len(moving_avg1)], 'c-', label=label4)

        plt.title('Rewards Horizon: discounted infinite vs undiscounted 6hr finite vs undiscounted 12hr finite')
        plt.ylabel('Energy Consumption for Summer Period 6/21 ~ 8/21')
        plt.xlabel('Episode')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    plot_horizon()
    #plot_hidden()
