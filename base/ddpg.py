import base2 as base
import matplotlib.pyplot as plt

import gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


device='cpu'

class replay_memory():
    def __init__(self,replay_memory_size):
        self.memory_size=replay_memory_size
        self.memory=np.array([])
        self.cur=0
        self.new=0
    def size(self):
        return self.memory.shape[0]
#[s,a,r,s_,done] make sure all info are lists, i.e. [[[1,2],[3]],[1],[0],[[4,5],[6]],[True]]
    def store(self,trans):
        if(self.memory.shape[0]<self.memory_size):
            if self.new==0:
                self.memory=np.array(trans)
                self.new=1
            elif self.memory.shape[0]>0:
                self.memory=np.vstack((self.memory,trans))
        else:
            self.memory[self.cur,:]=trans
            self.cur=(self.cur+1)%self.memory_size

    def sample(self,batch_size):
        if self.memory.shape[0]<batch_size:
            return -1
        sam=np.random.choice(self.memory.shape[0],batch_size)
        return self.memory[sam]

def gumbel_sample(shape,eps=1e-10):
    seed=th.FloatTensor(shape).uniform_().to(device)
    return -th.log(-th.log(seed+eps)+eps)

def gumbel_softmax_sample(logits,temperature=1.0):
    #print(logits)
    logits=logits+gumbel_sample(logits.shape,1e-10)
    #print(logits)
    return (th.nn.functional.softmax(logits/temperature,dim=1))

def gumbel_softmax(prob,temperature=1.0,hard=False):
    #print(prob)
    logits=th.log(prob)
    y=gumbel_softmax_sample(prob,temperature)
    if hard==True:   #one hot but differenttiable
        y_onehot=onehot_action(y)
        y=(y_onehot-y).detach()+y
    return y

def onehot_action(prob):
    y=th.zeros_like(prob).to(device)
    index=th.argmax(prob,dim=1).unsqueeze(1)
    y=y.scatter(1,index,1)
    return y.to(th.long)

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.fc1=nn.Linear(state_dim,170)
        self.fc2=nn.Linear(170+act_dim,100)
        self.fc3=nn.Linear(100,1)

    def forward(self,state,action):
        x=self.fc1(state)
        x=F.relu(x)
        x=self.fc2(th.cat((x,action),1))
        x=F.relu(x)
        x=self.fc3(x)
        return x

class Actor(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,170)
        self.fc2=nn.Linear(170,100)
        self.fc3=nn.Linear(100,act_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        return th.tanh(x)


default_args = {'idf': '../in.idf',
                'epw': '../weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }
lr=3e-4
tau=0.05
max_t=20000
gamma=0.97
memory_size=10000
batchsize=300
warmup=batchsize
env = base.EnergyPlusEnv(default_args)
device="cpu"
n_action=env.action_space.n
n_state=env.observation_space.shape[0]

class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action).to(device)
        self.target_actor=Actor(n_state,n_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lr)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lr)

    def choose_action(self,state,eps):
        prob=self.actor(th.FloatTensor(state).to(device))
        prob=th.nn.functional.softmax(prob,0)
        #print(prob)
        if np.random.uniform()>eps:
            action=th.argmax(prob,dim=0).tolist()
        else:
            action=np.random.randint(0,n_action)
        return action

    def actor_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)

        differentiable_a=th.nn.functional.gumbel_softmax(th.log(th.nn.functional.softmax(self.actor(b_s),dim=1)),hard=True)
        #print(differentiable_a)
        #differentiable_a2=th.nn.functional.softmax(th.nn.functional.softmax(self.actor(b_s),dim=1),dim=1)
        #index=th.argmax(differentiable_a2,dim=1).unsqueeze(1)
        #oh=th.zeros_like(differentiable_a2).scatter_(1,index,1)
        #differentiable_a2=(oh-differentiable_a2).detach()+differentiable_a2

        loss=-self.critic(b_s,differentiable_a).mean()
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()

    def critic_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.zeros(batchsize,n_action).scatter_(1,th.LongTensor(batch[:,2].tolist()),1).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        eval_q=self.critic(b_s,b_a)

        next_action=th.nn.functional.softmax(self.target_actor(b_s_),dim=1)

        index=th.argmax(next_action,dim=1).unsqueeze(1)
        next_action=th.zeros_like(next_action).scatter_(1,index,1).to(device)
        #print(next_action)
        target_q=th.zeros_like(eval_q).to(device)

        for i in range(b_d.shape[0]):
            target_q[i]=(1-b_d[i,0])*gamma*self.target_critic(b_s_,next_action)[i].detach()+b_r[i]
        td_error=eval_q-target_q
        loss=(td_error**2).mean()
        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()


    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)



def graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, episode):
    start = 310
    end = 1010
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


def save_reward(reward: float) -> None:
    f_name = './logs/ddpg-scores.txt'
    with open(f_name, 'a') as scores_f:
        scores_f.write(str(reward) + '\n')

for j in range(1):
    ddpg=DDPG()
    highest=0
    for episode in range(10000):
        s=env.reset()
        t=0
        total_reward=0

        cooling_setpoints = []
        cost_signals = []
        outdoor_temperatures = []
        indoor_temperatures = []
        while(t<max_t):
            a=ddpg.choose_action(s,0.1)
            s_,r,done,_, info=env.step(a)

            cooling_setpoints.append(info['cooling_actuator_value'])
            cost_signals.append(info['cost_signal'])
            outdoor_temperatures.append(s_[0])
            indoor_temperatures.append(s_[1])

            total_reward+=r
            transition=[s,[r],[a],s_,[done]]
            ddpg.memory.store(transition)
            if done:
                break
            s=s_
            if(ddpg.memory.size()<warmup):
                continue
            batch=ddpg.memory.sample(batchsize)
            ddpg.critic_learn(batch)
            ddpg.actor_learn(batch)
            ddpg.soft_update()

            t+=1

        save_reward(total_reward)
        graphing(cooling_setpoints, cost_signals, outdoor_temperatures, indoor_temperatures, episode)
