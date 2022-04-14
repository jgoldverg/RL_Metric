import gym
from gym import spaces
import numpy as np

class NetEnvironment(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)
        # clarification needed in defining action space: discrete?
        # fix a set of actions and pass a 'batch' of state with actions to net?
        self.action_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.int16)
    def reset(self):
        pass
    def step(self, observation):
        pass
    def render(self):
        pass
