import gym
from gym import spaces
import numpy as np
from fileData import environmentGroup
import random

THRUPUT_PENALTY = -0.3 # hyperparameter
MAX_TIMESTEPS = 100
MAX_ACTIONS = 127

"""
NetEnvironment Class is the gym environment for each environmentGroup
input: environment_group:   environment group for a key
group key: 'FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'
"""
class NetEnvironment(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, environment_group):
        self.environment_group = environment_group
        self.states = environment_group.return_state_list()
        self.actions = environment_group.return_action_list()
        self.max_throughput = environment_group.group_maximum_throughput()
        self.max_throughput_parameters=environment_group.return_group_max_throughput_parameters()
        self.environment_group_identification=environment_group.return_group_identification()
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_timesteps = MAX_TIMESTEPS
        self.time = 0
        self.b = THRUPUT_PENALTY
        self.prev_throughput = -1.
        self.current_observation = np.asarray(self.states[0])
        self.obs_shape=(8,)

    def reset(self):
        self.time = 0
        self.prev_throughput = -1
        self.current_observation = self.states[0]
        return np.asarray(self.current_observation)

    def step(self, action):
        action = self.actions[action]
        # action = tuple(action)
        # get throughputs
        throughputs = self.environment_group.return_group_key_throughput(action)
        cur_throughput = random.choice(throughputs)
        # if cur_throughput < self.prev_throughput:
        #     reward = self.b
        # else:
        reward = cur_throughput / self.max_throughput
        self.prev_throughput = cur_throughput

        self.time += 1
        if self.max_timesteps <= self.time:
            done = True
        else:
            done = False

        info = {'time': self.time, 'max_time': self.max_timesteps}
        self.current_observation[-3:] = action
        return np.asarray(self.current_observation), reward, done, info

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_max_throughput(self):
        return self.max_throughput

    def get_time(self):
        return self.time

    def render(self):
        pass

class NetEnvironment_glob(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self, environment_group):
        self.environment_group = environment_group
        self.states = environment_group.return_state_list()
        self.actions = environment_group.return_action_list()
        self.max_throughput = environment_group.group_maximum_throughput()
        self.max_throughput_parameters=environment_group.return_group_max_throughput_parameters()
        self.environment_group_identification=environment_group.return_group_identification()
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_timesteps = MAX_TIMESTEPS
        self.time = 0
        self.b = THRUPUT_PENALTY
        self.prev_throughput = -1.
        self.current_observation = np.asarray(self.states[0])
        self.obs_shape=(8,)

    def reset(self):
        self.time = 0
        self.prev_throughput = -1
        self.current_observation = self.states[0]
        return np.asarray(self.current_observation)

    def step(self, action):
        action = self.actions[action % self.action_space.n]
        # action = tuple(action)
        # get throughputs
        throughputs = self.environment_group.return_group_key_throughput(action)
        cur_throughput = random.choice(throughputs)
        # if cur_throughput < self.prev_throughput:
        #     reward = self.b
        # else:
        reward = cur_throughput / self.max_throughput
        self.prev_throughput = cur_throughput

        self.time += 1
        if self.max_timesteps <= self.time:
            done = True
        else:
            done = False

        info = {'time': self.time, 'max_time': self.max_timesteps}
        self.current_observation[-3:] = action
        return np.asarray(self.current_observation), reward, done, info

    def get_actions(self):
        return self.actions

    def get_states(self):
        return self.states

    def get_max_throughput(self):
        return self.max_throughput

    def get_time(self):
        return self.time

    def render(self):
        pass
    

class GroupNetWrapper(gym.Env):
    metadata = metadata = {'render.modes': []}
    def __init__(self):
        self.env_list = []
        self.cur_env = None
        self.probabilities = []
        self.p = 1/3 # probability
        self.r = 1. # isolation constant
        self.obs_shape=(8,)
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        
        # context
        self.observation_space = None
        self.max_throughput = None
        self.max_throughput_parameters=None
        self.environment_group_identification=None

    def reset(self):
        for env in self.env_list:
            env.reset()
        if self.cur_env != None:
            return self.cur_env.reset()
        else:
            return None # something went wrong
    
    def reset_list(self):
        self.env_list = [] # could replace with a pointer
        self.probabilities = []
        self.cur_env = None
        
    def add(self, env):
        if len(self.env_list) == 0:
            self.env_list.append(env)
            self.probabilities.append(1.)
            self.cur_env = env
            self.observation_space = env.observation_space
            self.max_throughput = env.max_throughput
            self.max_throughput_parameters=env.max_throughput_parameters
            self.environment_group_identification=env.environment_group_identification
        else:
            self.probabilities[-1] = self.r * (1 - self.p)
            self.env_list.reverse()
            self.env_list.append(env)
            self.env_list.reverse()
            self.probabilities.append(self.r * self.p)
            self.r *= self.p
            slice_list = self.env_list[1:]
            np.random.shuffle(slice_list)
            self.env_list[1:] = slice_list
    
    def choose_env(self):
        if len(self.env_list) == 0:
            return None
        nparray = np.array(self.env_list, dtype=object)
        self.cur_env = np.random.choice(nparray, p=self.probabilities)
        self.observation_space = self.cur_env.observation_space
        self.action_space = self.cur_env.action_space
        self.observation_space = self.cur_env.observation_space
        self.max_throughput = self.cur_env.max_throughput
        self.max_throughput_parameters=self.cur_env.max_throughput_parameters
        self.environment_group_identification=self.cur_env.environment_group_identification
        return self.cur_env
        
    def step(self, action, change=True):
        next_ob, reward, done, info = self.cur_env.step(action)
        if change:
            self.cur_env = self.choose_env()
        return next_ob, reward, done, {'rotation': np.asarray(self.cur_env.current_observation)}

    def render(self):
        pass

import torch
from torch import nn
import gym
from collections import deque
import itertools
import random
import matplotlib.pyplot as plt

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=50000
MIN_REPLAY_SIZE=10000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=100000
TARGET_UPDATE_FREQ=1000

def print_list_action(lst,env):
  action_dictionary=dict()
  for action in range(env.action_space.n):
    value=env.actions[action]
    action_dictionary[action]=value
  return_list=[]
  for i in lst:
      return_list.append(action_dictionary[i])
  return return_list


class Network_glob(nn.Module):
    def __init__(self,env):
        super().__init__()
        in_features=int(np.prod(env.obs_shape))
        self.net=nn.Sequential(
                nn.Linear(in_features,128),
            nn.Tanh(),
            nn.Linear(128, env.action_space.n))

    def forward(self,x):
         return self.net(x)

    def act(self,obs):
        obs_t=torch.as_tensor(obs,dtype=torch.float32)
        q_values=self(obs_t.unsqueeze(0))
        max_q_index=torch.argmax(q_values,dim=1)[0]
        action=max_q_index.detach().item()

        return action

class DQNAgent_glob():
  def __init__(self,env,GAMMA=0.99,
                BATCH_SIZE=32,
                BUFFER_SIZE=50000,
                MIN_REPLAY_SIZE=10000,
                EPSILON_START=1.0,
                EPSILON_END=0.02,
                EPSILON_DECAY=100000,
                TARGET_UPDATE_FREQ=1000):
    self.env=env
    self.GAMMA=GAMMA
    self.BATCH_SIZE=BATCH_SIZE
    self.BUFFER_SIZE=BUFFER_SIZE
    self.MIN_REPLAY_SIZE=MIN_REPLAY_SIZE
    self.EPSILON_START=EPSILON_START
    self.EPSILON_DECAY=EPSILON_DECAY
    self.EPSILON_END=EPSILON_END
    self.TARGET_UPDATE_FREQ=TARGET_UPDATE_FREQ
    self.replay_buffer=deque(maxlen=self.BUFFER_SIZE)
    self.rew_buffer=deque([0.0],maxlen=100)
    self.episode_reward=0.0
    self.online_net=Network_glob(env)
    self.target_net=Network_glob(env)
    self.target_net.load_state_dict(self.online_net.state_dict())
    self.optimizer=torch.optim.Adam(self.online_net.parameters(),lr=5e-4)
    self.reward_per_episode=[]
    self.epsilon_per_episode=[]

  def warming_replay_buffer(self):
    obs=self.env.reset()
    for _ in range(self.MIN_REPLAY_SIZE):
      action=self.env.action_space.sample()
      new_obs,rew,done, info =self.env.step(action)
      transition=(obs,action,rew,done,new_obs)
      self.replay_buffer.append(transition)
      obs=info['rotation']
      if done:
          obs=self.env.reset()


  def training_endlessly(self):
    obs=self.env.reset()
    for step in itertools.count():
    #for step in range(TRAINING_STEPS):
        epsilon=np.interp(step,[0,self.EPSILON_DECAY],[self.EPSILON_START,self.EPSILON_END])
        rnd_sample=random.random()

        ####following implements epsilon-greedy policy
        if rnd_sample <=epsilon:
            action=self.env.action_space.sample()
        else:
            action=self.online_net.act(obs)

        new_obs,rew,done, _ =self.env.step(action)
        transition= (obs,action,rew,done,new_obs)
        self.replay_buffer.append(transition)
        obs=new_obs
        self.episode_reward+=rew
        if (step % 100 == 0):
            obs=self.env.reset()
            self.rew_buffer.append(self.episode_reward)
            self.reward_per_episode.append(self.episode_reward)
            self.epsilon_per_episode.append(epsilon)
            self.episode_reward=0.0

        #### Satrt gradient Step
        transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
    #     print(transitions)
        obses=np.asarray([t[0] for t in transitions])
        actions=np.asarray([t[1] for t in transitions])
        rews=np.asarray([t[2] for t in transitions])
        dones=np.asarray([t[3] for t in transitions])
        new_obses=np.asarray([t[4] for t in transitions])

        obses_t=torch.as_tensor(obses,dtype=torch.float32)
        actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
        rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
        dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
        new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)

        #compute Targets
        target_q_values=self.target_net(new_obses_t)
        max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
        targets=rews_t+GAMMA *(1-dones_t) * max_target_q_values

        # Compute Loss
        q_values=self.online_net(obses_t)
        action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
        loss=nn.functional.smooth_l1_loss(action_q_values,targets)

        ## Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # update target network
        if step % self.TARGET_UPDATE_FREQ ==0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        ##logging
        if step %1000 ==0:
            print()
            print('step', step)
            print('Avg Rew',np.mean(self.rew_buffer))

  def training(self,TRAINING_STEPS=170000):
    obs=self.env.reset()
    # for step in itertools.count():
    for step in range(TRAINING_STEPS):
        epsilon=np.interp(step,[0,self.EPSILON_DECAY],[self.EPSILON_START,self.EPSILON_END])
        rnd_sample=random.random()

        ####following implements epsilon-greedy policy
        if rnd_sample <=epsilon:
            action=self.env.action_space.sample()
        else:
            action=self.online_net.act(obs)

        new_obs,rew,done, info =self.env.step(action)
        transition= (obs,action,rew,done,new_obs)
        self.replay_buffer.append(transition)
        obs=info['rotation']
        self.episode_reward+=rew
        if (step % 100 == 0):
            obs=self.env.reset()
            self.rew_buffer.append(self.episode_reward)
            self.reward_per_episode.append(self.episode_reward)
            self.epsilon_per_episode.append(epsilon)
            self.episode_reward=0.0

        #### Satrt gradient Step
        transitions=random.sample(self.replay_buffer, self.BATCH_SIZE)
    #     print(transitions)
        obses=np.asarray([t[0] for t in transitions])
        actions=np.asarray([t[1] for t in transitions])
        rews=np.asarray([t[2] for t in transitions])
        dones=np.asarray([t[3] for t in transitions])
        new_obses=np.asarray([t[4] for t in transitions])

        obses_t=torch.as_tensor(obses,dtype=torch.float32)
        actions_t=torch.as_tensor(actions,dtype=torch.int64).unsqueeze(-1)
        rews_t=torch.as_tensor(rews,dtype=torch.float32).unsqueeze(-1)
        dones_t=torch.as_tensor(dones,dtype=torch.float32).unsqueeze(-1)
        new_obses_t=torch.as_tensor(new_obses,dtype=torch.float32)

        #compute Targets
        target_q_values=self.target_net(new_obses_t)
        max_target_q_values=target_q_values.max(dim=1,keepdim=True)[0]
        targets=rews_t+GAMMA *(1-dones_t) * max_target_q_values

        # Compute Loss
        q_values=self.online_net(obses_t)
        action_q_values=torch.gather(input=q_values,dim=1,index=actions_t)
        loss=nn.functional.smooth_l1_loss(action_q_values,targets)

        ## Gradient Descent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        # update target network
        if step % self.TARGET_UPDATE_FREQ ==0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        ##logging
        if step %1000 ==0:
            print()
            print('step', step)
            print('Avg Rew',np.mean(self.rew_buffer))

  def save_model(self):
    name='agent'+str(self.env.environment_group_identification)+'yo'
    torch.save(self.online_net, name+"online_net")
    torch.save(self.target_net, name+"target_net")

  def load_model(self,online_net,target_net):
    self.online_net = torch.load("online_net")
    self.target_net = torch.load("target_net")

  def plot_training_curve(self):
    reward_epsilon_values=[]
    reward_epsilon_values.append(self.reward_per_episode)
    reward_epsilon_values.append(self.epsilon_per_episode)
    labels = ["reward", "epsilon"]
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('reward-episodes & epsilon-episodes graph DQN for selectedGroup')
    for i,ax in enumerate(axs):
        axs[i].plot(reward_epsilon_values[i],label=labels[i])
        axs[i].legend(loc="lower right")
    plt.show()

  def validation(self, env, TOTAL_EPISODES_VALIDATION=10):
    self.reward_per_episode_validation=[]
    self.action_list_per_episode=[]
    for episode in range(TOTAL_EPISODES_VALIDATION):
        obs=env.reset()
        done=False
        episode_reward=0
        action_list=[]
        while(done==False):
            action=self.online_net.act(obs)
            action_list.append(action)
            new_obs,rew,done, _ =env.step(action)
            episode_reward+=rew
            obs=new_obs
        self.reward_per_episode_validation.append(episode_reward)
        self.action_list_per_episode.append(action_list)
    return self.reward_per_episode_validation,self.action_list_per_episode

  def plot_validation_curve(self):
    reward_epsilon_values=[]
    reward_epsilon_values.append(self.reward_per_episode_validation)
    # reward_epsilon_values.append(self.reward_per_episode_validation)
    labels = ["reward", "reward"]
    fig, axs = plt.subplots(2, sharex=True)
    fig.suptitle('reward-episodes & reward-episodes graph DQN for 10 episode of validation for the group')
    for i,ax in enumerate(axs):
        axs[i].plot(reward_epsilon_values[i],label=labels[i])
        axs[i].legend(loc="lower right")
    plt.show()


  def print_action_list_for_validation_episodes(self,action_list):
    for i in range(len(action_list)):
      print("Validation episode ",i, " actions taken")
      print(print_list_action(action_list[i],self.env))

