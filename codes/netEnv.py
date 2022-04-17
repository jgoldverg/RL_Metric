import gym
from gym import spaces
import numpy as np
from fileData import environmentGroup
import random

THRUPUT_PENALTY = -0.3 # hyperparameter
MAX_TIMESTEPS = 100

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
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.actions))
        
        self.max_timesteps = MAX_TIMESTEPS
        self.time = 0
        self.b = THRUPUT_PENALTY
        self.prev_throughput = -1.
        self.current_observation = self.states[0]
        
    def reset(self):
        self.time = 0
        self.prev_throughput = -1
        self.current_observation = self.states[0]
        return self.current_observation
    
    def step(self, action):
        action = tuple(action)
        # get throughputs
        throughputs = self.environment_group.return_group_key_throughput(action)
        cur_throughput = random.choice(throughputs)
        if cur_throughput < self.prev_throughput:
            reward = self.b
        else:
            reward = cur_throughput / self.max_throughput
        self.prev_throughput = cur_throughput
        
        self.time += 1
        if self.max_timesteps <= self.time:
            done = True
        else:
            done = False
        
        info = {'time': self.time, 'max_time': self.max_timesteps}
        self.current_observation[-3:] = action
        return self.current_observation, reward, done, info
    
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
