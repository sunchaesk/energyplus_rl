
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import json


import base as base
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

def plot_penalty():
    data_fname1 = './logs/scores-base-128.txt'
    data_fname2 = './logs/scores-penalty-0.9-128.txt'
    plt.figure(figsize=(30,5))
    label1 = 'no penalty'
    label2 = 'penalty'
    with open(data_fname1, 'r') as data1, open(data_fname2, 'r') as data2:
        scores1 = data1.read().split('\n')
        scores2 = data2.read().split('\n')
        del scores1[-1]
        del scores2[-1]
        scores1 = np.array([-float(x) for x in scores1])
        scores2 = np.array([-float(x) for x in scores2])
        scores_idx1 = np.array(list(range(len(scores1))))
        scores_idx2 = np.array(list(range(len(scores2))))
        plt.plot(scores_idx1, scores1, 'b-', label=label1)
        plt.plot(scores_idx2, scores2, 'r-', label=label2)
        plt.title('Penalty VS No Penalty / 128 hidden, gamma=0.9')
        plt.ylabel('Energy Consumption for Summer Period 6/21 ~ 8/21 (J)')
        plt.xlabel('Episode')
        plt.legend()
        plt.show()

        # window = 5
        # moving_avg1 = np.convolve(scores1, np.ones(window)/window, mode='valid')
        # moving_avg2 = np.convolve(scores2, np.ones(window)/window, mode='valid')
        # plt.plot(range(window - 1, len(scores1)), moving_avg1, 'r-', label=label1)
        # plt.plot(range(window - 1, len(scores2)), moving_avg2, 'b-', label=label2)
        # plt.title(f'Penalty VS No Penalty ({window} ep Moving Average)')
        # plt.ylabel('Energy Consumption for Summer Period 6/21 ~ 8/21 (J)')
        # plt.xlabel('Episode')
        # plt.legend()
        # plt.show()

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

def plot_file(fname,x,y,title):
    window = 5
    with open(fname, 'r') as data:
        plt.figure(figsize=(30,5))
        scores = data.read().split('\n')
        del scores[-1]
        scores = np.array([-float(x) for x in scores])
        scores_idx = np.array(list(range(len(scores))))
        plt.plot(scores_idx, scores, 'r-')

        # moving_avg1 = np.convolve(scores, np.ones(window)/window, mode='valid')
        # plt.plot(range(window - 1,len(scores)), moving_avg1, 'b-')

        # moving_avg2 = np.convolve(scores, np.ones(10)/10, mode='valid')
        # plt.plot(range(10 - 1, len(scores)), moving_avg2, 'g-')

        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        plt.show()

def plot_list(d_list, style='fit'):
    '''
    two availabe styles: fit, all
    - fit fits all graphs to the min data len
    - all fits all data to their respective data len
    '''
    fs = []
    data_list = []
    try:
        open_file_list = [f[0] for f in d_list]
        for f in open_file_list:
            fs.append(open(f, 'r'))
        for ff in fs:
            data_list.append(ff.read().split('\n')[:-1])

        plt.figure(figsize=(25,5))

        if style == 'fit':
            min_dlen = min([len(d) for d in data_list])
            min_x = list(range(min_dlen))
            for i in range(len(data_list)):
                for j in range(len(data_list[i])):
                    data_list[i][j] = -float(data_list[i][j])
                plt.plot(min_x,data_list[i][0:min_dlen], label=d_list[i][1])

        if style == 'all':
            for i in range(len(data_list)):
                for j in range(len(data_list[i])):
                    data_list[i][j] = -float(data_list[i][j])
                x_vals = list(range(len(data_list[i])))
                plt.plot(x_vals, data_list[i], label=d_list[i][1])


        #plt.axhline(146025024.35006934, color='red', linestyle='dashdot', linewidth=5)
        plt.legend()
        plt.show()
    finally:
        for f in fs:
            f.close()

def caps_graphing():
    # Create a figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    episode_reward_1 = [-4442.75027287003, -4387.380989430237, -4406.058142905367, -4556.337650839784, -4529.680958223968, -4509.6499448361155, -4480.636283058975, -4570.903686875725, -4615.5949552773545, -4650.649696694949, -4619.36459392848, -4353.5093821791515, -4424.357179697982, -4460.5912846188385, -4458.071425147296, -4420.0424937700855, -4406.1362831425195, -4422.315664622684, -4432.9308901262475, -4618.526694900249]
    episode_reward_2 = [-4603.51834112976, -4636.271030621992, -4537.322324695106, -4600.801211216524, -4615.8681067322095, -4613.500405621157, -4568.581537224182, -4546.434044398001, -4634.932267799854, -4606.713925292639, -4616.800367474106, -4609.722648410097, -4625.375934224491, -4555.5714056181505, -4596.929934035622, -4645.394464774823, -4624.952375257118, -4628.3279418250695, -4603.26301379163, -4611.770125415244]
    episode_reward_3 = [-4663.860405525254, -4661.753008940904, -4709.384646012639, -4706.660664151698, -4683.289345111921, -4720.685453013332, -4706.117014748142, -4712.480514527388, -4722.2749384406725, -4714.511674178798, -4773.372150240414, -4739.450300112793, -4641.219389272806, -4721.06506644994, -4611.290237468562, -4635.040970580297, -4742.539659750439, -4695.869212057943, -4678.338689707195, -4676.508745324079]
    episode_reward_4 = [-4457.931503572588, -4389.2699909606135, -4369.883885145997, -4427.402318557886, -4397.969238200458, -4423.98951967602, -4398.8805500232, -4424.231533667353, -4454.239655933743, -4348.904795957971, -4414.327218306235, -4438.055852771265, -4420.820683770996, -4459.798640644178, -4441.053014151909, -4404.283536098049, -4471.583688376114, -4452.255245021906, -4477.293990669138, -4425.46122343037]
    episode_reward_5 = [-4418.907431002812, -4439.300031638898, -4403.1587876322465, -4444.555254213128, -4375.706416120955, -4424.761727860595, -4392.788557988743, -4420.390707918448, -4438.670162547734, -4426.419747218515, -4438.048840232787, -4458.9141433483865, -4411.117256991809, -4407.499946347132, -4423.29592819042, -4435.234731735033, -4426.925658830048, -4370.2110135336015, -4381.382393915314, -4383.70380143835]
    for i in range(len(episode_reward_1)):
        episode_reward_1[i] *= -1
        episode_reward_2[i] *= -1
        episode_reward_3[i] *= -1
        episode_reward_4[i] *= -1
        episode_reward_5[i] *= -1

    total_variance_1 = [931.9000000000017, 915.7000000000016, 896.3000000000019, 926.4000000000013, 919.0000000000015, 918.4000000000013, 931.6000000000017, 1086.8000000000013, 1102.8000000000022, 1107.4000000000017, 1098.6000000000022, 1025.2000000000025, 1042.600000000002, 1030.6000000000026, 1056.6000000000024, 884.8000000000029, 870.0000000000026, 911.000000000003, 911.6000000000031, 823.5000000000023]
    total_variance_2 = [730.7000000000014, 727.1000000000014, 731.1000000000017, 732.5000000000013, 729.5000000000014, 717.5000000000015, 712.7000000000014, 737.5000000000013, 729.7000000000014, 734.9000000000015, 724.9000000000017, 719.3000000000015, 730.3000000000014, 742.5000000000016, 737.3000000000013, 714.1000000000015, 711.9000000000015, 703.9000000000016, 724.9000000000017, 727.7000000000014]
    total_variance_3 = [948.9000000000015, 931.300000000001, 947.3000000000006, 930.3000000000009, 942.5000000000009, 937.100000000001, 944.900000000001, 963.5000000000008, 937.3000000000011, 930.9000000000009, 944.3000000000011, 916.500000000001, 909.3000000000011, 924.5000000000013, 898.9000000000011, 924.1000000000009, 925.1000000000007, 957.1000000000008, 942.100000000001, 897.7000000000008]
    total_variance_4 = [884.4000000000015, 866.6000000000018, 884.4000000000013, 858.0000000000014, 843.2000000000013, 878.6000000000015, 886.8000000000013, 896.4000000000009, 865.0000000000016, 882.6000000000016, 877.8000000000013, 881.2000000000018, 882.4000000000011, 856.4000000000013, 861.4000000000013, 853.4000000000012, 849.0000000000015, 875.2000000000016, 852.6000000000017, 874.4000000000013]
    total_variance_5 = [738.8000000000014, 756.0000000000014, 749.8000000000014, 756.4000000000013, 772.2000000000014, 759.8000000000014, 768.8000000000012, 749.8000000000017, 734.6000000000014, 745.2000000000013, 755.6000000000014, 745.6000000000016, 760.4000000000013, 765.4000000000015, 736.2000000000013, 778.4000000000015, 768.8000000000014, 744.2000000000013, 753.0000000000013, 760.2000000000013]

    x = list(range(20))

    # Plot on the top subplot
    ax1.plot(x, episode_reward_1, label='No CAPS')
    ax1.plot(x, episode_reward_2, label='CAPS lambda_a = 1, lambda_s = 5')
    ax1.plot(x, episode_reward_3, label='CAPS lambda_a = 3, lambda_s = 5')
    ax1.plot(x, episode_reward_4, label='CAPS lambda_a = 1, lambda_s = 10')
    ax1.plot(x, episode_reward_5, label='CAPS lambda_a = 1, lambda_s = 2')
    ax1.set_title('Episode Rewards')
    ax1.legend()

    # Plot on the bottom subplot
    ax2.plot(x, total_variance_1, label='No CAPS')
    ax2.plot(x, total_variance_2, label='CAPS lambda_a = 1, lambda_s = 5')
    ax2.plot(x, total_variance_3, label='CAPS lambda_a = 3, lambda_s = 5')
    ax2.plot(x, total_variance_4, label='CAPS lambda_a = 1, lambda_s = 10')
    ax2.plot(x, total_variance_5, label='CAPS lambda_a = 1, lambda_s = 2')
    ax2.set_title('Total Variance')
    ax2.legend()

    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the figure
    plt.show()

# WITHOUT CAPS

# WITH CAPS lambda_a = 1

# WITH CAPS lambda_a = 3

if __name__ == "__main__":
    caps_graphing()
    sys.exit(1)
    print(len(sys.argv))
    if len(sys.argv) < 2:
        print('default')
        plot_file('./logs/scores.txt', x='episode', y='Episode Energy Consumption (6,21) ~ (6,28) (J)', title='demand response training episodes')
    else:
        print('custom')
        f_name = './logs/' + sys.argv[1] + '.txt'
        plot_file(f_name, x='episode', y='Episode Energy Consumption', title='demand response training file: {} [30 timestep forecasts interval of 3]'.format(sys.argv[1]))

    # l = [
    #     ('./logs/scores.txt', 'no dr'),
    #     ('./logs/dr-fail.txt', 'dr'),
    # ]
    # plot_list(l, style='all')
    #plot_penalty()
    #plot_horizon()
    #plot_hidden()
