

import base

import os
import random

import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2500) # self.replay is hit when memory full
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(Dense(48, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    # TODO: debug
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        print('REPLAY HIT')
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            print('hit')
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def load_prev_model(agent: DQNAgent):
    '''
    NOTE: deprecate -> just do this stuff manually
    '''
    prev_models = os.listdir('./models/')
    prev_models = [os.path.splitext(fname)[0] for fname in prev_models]

default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2
                }

if __name__ == "__main__":
    env = base.EnergyPlusEnv(default_args)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)


    batch_size = 128
    checkpoint_num = 20

    scores = np.array([])
    episodes = 500
    start_episode = 0

    try:
        scores = np.genfromtxt('saved_scores.csv', delimiter=',')
        start_episode = len(scores)
        agent.load('./model/agent-{}'.format(start_episode))
    except:
        print('## saved_scores.csv empty')

    #NOTE: DEPRECATED LOAD stuff, comment/uncomment to load model or not
    #curr_episode, prev_scores = load_prev_model(agent)

    print('## state_size', state_size, '## action_size', action_size)
    print('## action_space', env.action_space)
    for episode in range(start_episode, episodes + 1):
        print('################################################')
        print('EPISODE:', episode)
        print('################################################')
        if episode % checkpoint_num == 0 and episode != 0:
            agent.save('./model/agent-{}'.format(episode))
            np.savetxt('saved_scores.csv', scores, delimiter=',')

        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        cumm_score = 0
        done = False
        while not done:
            print('## EPSILON:', agent.epsilon)
            action = agent.act(state)
            next_state, reward, done, truncated, info = env.step(action) # truncated is not in use
            reward *= -1 # negate it to minimize
            cumm_score += reward
            #next_state = np.reshape(next_state, [1,state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print("episode: {}/{}, score:{}, e:{:.2}".format(episode, episodes,cumm_score,agent.epsilon))
        scores = np.append(scores, [cumm_score * -1])
        #scores.append(cumm_score * -1)

    plt.plot(scores)
    plt.ylabel('energy consumption')
    plt.xlabel('episodes')
    plt.title('E+ Reinforcement Learning')
    plt.show()

#episode: 21/500, score:-81285147256.42249, e:1.0
# TODO
# - remove print statement -> just show current episode
# - make a way to load model and still generate the graph at the end
# - store the scores array somehow

    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # agent = DQNAgent(state_size, action_size)
    # # agent.load("./save/cartpole-dqn.h5")
    # done = False
    # batch_size = 32

    # for e in range(EPISODES):
    #     state = env.reset()
    #     state = np.reshape(state, [1, state_size])
    #     for time in range(500):
    #         # env.render()
    #         action = agent.act(state)
    #         next_state, reward, done, _ = env.step(action)
    #         reward = reward if not done else -10
    #         next_state = np.reshape(next_state, [1, state_size])
    #         agent.memorize(state, action, reward, next_state, done)
    #         state = next_state
    #         if done:
    #             print("episode: {}/{}, score: {}, e: {:.2}"
    #                   .format(e, EPISODES, time, agent.epsilon))
    #             break
    #         if len(agent.memory) > batch_size:
    #             agent.replay(batch_size)

    # print('main')
    # env = base.EnergyPlusEnv(default_args)
    # print('action_space:', end='')
    # print(env.action_space)
    # scores = []
    # for episode in range(2):
    #     state = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         #env.render()
    #         # action = env.action_space.sample()
    #         #action = 22.0
    #         ret = n_state, reward, done, info, STUFF = env.step(0)
    #         #print('RET STUFF:', ret)
    #         score+=reward
    #         # print('DONE?:', done)
    #         print('Episode:{} Reward:{} Score:{}'.format(episode, reward, score))

    #     scores.append(score)
    # print("SCORES: ", scores)
    # print("TRULY DONE?") # YES, but program doesn't terminate due to threading stuff?
