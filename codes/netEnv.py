import gym
from gym import spaces
import numpy as np

class NetEnvironment(gym.Env):
    metadata = {'render.modes': []}
    def __init__(self):
        # unbounded box for now
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.int16)
    def reset(self):
        pass
    def step(self, observation):
        pass
    def render(self):
        pass
