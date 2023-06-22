
import sys

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

import base_cont as base


OUTDOOR_TEMP = 0
RELATIVE_HUMIDITY = 3
MEAN_RADIANT = 2

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        #print('x', x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, inputs, labels, num_epochs, learning_rate):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                'start_date': (6,21), # DEPRECATED -> fixed the idf running problem
                'end_date': (8,21),
                'pmv_pickle_available': True,
                'pmv_pickle_path': './pmv_cache.pickle'
                }

def test_model(model, variable_index):
    '''
    @param model: pytorch model
    @param variable_index: index within the observation vector
    '''
    env = base.EnergyPlusEnv(default_args)
    state = env.reset()
    hour = 0
    minute = 0
    done = False

    guess = []
    actual = []

    window_size = 10
    prev = deque(maxlen=10)

    while not done:
        action = 0

        if len(prev) < prev.maxlen:

            ret = n_state, reward, done, truncated, info = env.step([action])
            prev.append(n_state[variable_index])
        else:

            input_tensor = torch.tensor([state[variable_index], hour, minute], dtype=torch.float64)
            input_tensor = input_tensor.double()
            #print(input_tensor)
            output = model(input_tensor)

            ret = n_state, reward, done, truncated, info = env.step([action])
            hour = info['hour']
            minute= info['minute']

            print('OUTPUT VS ACTUAL', output, info['obs_vec'][variable_index])
            guess.append(output.detach().numpy())
            actual.append(np.interp(n_state[0], [-1, 1], [15, 30]))
    #
    x = list(range(len(guess)))
    plt.plot(x, guess, 'r-')
    plt.plot(x, actual, 'b-')
    plt.show()
    print('DONE')

def training_data(window_size):
    env = base.EnergyPlusEnv(default_args)


    outdoor_temp_data = []
    relative_humidity_data = []
    mean_radiant_data = []
    hour_data = []
    minute_data = []

    state = env.reset()
    done = False
    while not done:
        action = 0
        ret = n_state, reward, done, truncated, info = env.step([action])

        outdoor_temp_data.append(info['obs_vec'][OUTDOOR_TEMP])
        relative_humidity_data.append(info['obs_vec'][RELATIVE_HUMIDITY])
        mean_radiant_data.append(info['obs_vec'][MEAN_RADIANT])
        hour_data.append(info['hour'])
        minute_data.append(info['minute'])

    # return {
    #     'outdoor_temp_data': outdoor_temp_data,
    #     'relative_humidity_data': relative_humidity_data,
    #     'mean_radiant_data': mean_radiant_data,
    #     'hour_data': hour_data,
    #     'minute_data': minute_data
    # }
    outdoor_temp_inputs = torch.tensor([outdoor_temp_data[i:i+window_size] for i in range(len(outdoor_temp_data)-window_size)], dtype=torch.float32).double()
    outdoor_temp_labels = torch.tensor(outdoor_temp_data[window_size:], dtype=torch.float32).unsqueeze(1).double()
    relative_humidity_inputs = torch.tensor([relative_humidity_data[i:i+window_size] for i in range(len(relative_humidity_data)-window_size)], dtype=torch.float32).double()
    relative_humidity_labels = torch.tensor(relative_humidity_data[window_size:], dtype=torch.float32).unsqueeze(1).double()
    mean_radiant_inputs = torch.tensor([mean_radiant_data[i:i+window_size] for i in range(len(mean_radiant_data)-window_size)], dtype=torch.float32).double()
    mean_radiant_labels = torch.tensor(mean_radiant_data[window_size:], dtype=torch.float32).unsqueeze(1).double()
    hour_inputs = torch.tensor(hour_data[window_size:], dtype=torch.float32).unsqueeze(1).double()
    minute_inputs = torch.tensor(minute_data[window_size:], dtype=torch.float32).unsqueeze(1).double()

    outdoor_temp_inputs = torch.cat((outdoor_temp_inputs, hour_inputs, minute_inputs), dim=1).double()
    relative_humidity_inputs = torch.cat((relative_humidity_inputs, hour_inputs, minute_inputs), dim=1).double()
    mean_radiant_inputs = torch.cat((mean_radiant_inputs, hour_inputs, minute_inputs), dim=1).double()


    return {
        "labels": {
            "outdoor_temp": outdoor_temp_labels,
            "relative_humidity": relative_humidity_labels,
            "mean_radiant": mean_radiant_labels
        },
        "inputs": {
            "outdoor_temp": outdoor_temp_inputs,
            "relative_humidity": relative_humidity_inputs,
            "mean_radiant": mean_radiant_inputs,
            # "hour": hour_inputs,
            # "minute": minute_inputs
        }
    }



# Define input and output

window_size = 10
input_size = window_size + 2
output_size = 1

hidden_size = 80

#model = MLP(input_size, hidden_size, output_size)

num_epochs = 10000
learning_rate = 0.001

#print(training_data(window_size))

if __name__ == "__main__":
    training_data_dict = training_data(window_size)
    print(training_data_dict)

    outdoor_temp_model = MLP(input_size, hidden_size, output_size)
    outdoor_temp_model.double()
    train(outdoor_temp_model,
          training_data_dict['inputs']['outdoor_temp'][:-10],
          training_data_dict['labels']['outdoor_temp'][:-10],
          num_epochs,
          learning_rate)

    relative_humidity_model = MLP(input_size, hidden_size, output_size)
    relative_humidity_model.double()
    train(relative_humidity_model,
          training_data_dict['inputs']['relative_humidity'][:-10],
          training_data_dict['labels']['relative_humidity'][:-10],
          num_epochs,
          learning_rate)

    mean_radiant_model = MLP(input_size, hidden_size, output_size)
    mean_radiant_model.double()
    train(mean_radiant_model,
          training_data_dict['inputs']['mean_radiant'][:-10],
          training_data_dict['labels']['mean_radiant'][:-10],
          num_epochs,
          learning_rate)

    save = False
    if save:
        torch.save({
            'num_epochs': num_epochs,
            'model': mean_radiant_model.state_dict()
        }, './model/forecast/mean-radiant.pt')

        torch.save({
            'num_epochs': num_epochs,
            'model': relative_humidity_model.state_dict()
        }, './model/forecast/relative-humidity.pt')

        torch.save({
            'num_epochs': num_epochs,
            'model': outdoor_temp_model.state_dict()
        }, './model/forecast/outdoor-temp.pt')

    test_model(mean_radiant_model, MEAN_RADIANT)
