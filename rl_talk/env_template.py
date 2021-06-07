from gym import Env
import numpy as np


class GridWorldEnv(Env):

    def __init__(self):
        self.state = np.zeros((3, 3, 3))

    def reset(self):
        self.state = np.zeros((3, 3, 3))
        return self.state

    def step(self):
        next_state, reward, is_done, info = None, None, None, None
        return next_state, reward, is_done, info

    def render(self):
        print(self.state)


