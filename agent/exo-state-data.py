
import sys

import base_cont as base

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

OUTDOOR_TEMP = 0 # exo
INDOOR_RELATIVE_HUMIDITY = 3 # not exo
MEAN_RADIANT = 2 # not exo
DIRECT_SOLAR = 6 # exo
HORIZONTAL_INFRARED = 7 # exo
OUTDOOR_RELATIVE_HUMIDITY = 8 # exo

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

def collect_data():
    outdoor_temp_data = []
    direct_solar_data = []
    horizontal_infrared_data = []
    outdoor_relative_humidity_data = []
    time_data = []

    env = base.EnergyPlusEnv(default_args)
    state = env.reset()
    done = False
    while not done:
        action = 0 # collecting exo states that are not dependent on action (indoor setpoints)

        ret = n_state, reward, done, truncated, info = env.step([action])

        outdoor_temp_data.append(info['obs_vec'][OUTDOOR_TEMP])
        direct_solar_data.append(info['obs_vec'][DIRECT_SOLAR])
        horizontal_infrared_data.append(info['obs_vec'][HORIZONTAL_INFRARED])
        outdoor_relative_humidity_data.append(info['obs_vec'][OUTDOOR_RELATIVE_HUMIDITY])

        current_time = tuple([
            info['year'],
            info['month'],
            info['day'],
            info['hour'],
            info['minute']
        ])
        time_data.append(current_time)

        state = n_state

    ret_dict = dict()
    for i in range(len(time_data)):
        ret_dict[time_data[i]] = {
            'outdoor_temp': outdoor_temp_data[i],
            'direct_solar': direct_solar_data[i],
            'horizontal_infrared': horizontal_infrared_data[i],
            'outdoor_relative_humidity': outdoor_relative_humidity_data[i]
        }
    print(ret_dict)

    save = True
    if save:
        with open('./exo-state.pt', 'wb') as handle:
            pickle.dump(ret_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    collect_data()
