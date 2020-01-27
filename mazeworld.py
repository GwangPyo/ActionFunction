import gym
import numpy as np


class MazeWorld(object):
    def __init__(self, shape):
        self.shape = shape
        assert shape[0] > 6 and shape[1] > 6, "World is too small"

    @staticmethod
    def distance(x, y):
        return np.abs(x[0] - y[0]) + np.abs(x[1] + y[1])

    @staticmethod
    def random_pos(max_x, max_y, min_x=0, min_y=0):
        x_pos = np.random.randint(low=min_x, high=max_x)
        y_pos = np.random.randint(low=min_y, high=max_y)
        return x_pos, y_pos

    def _build_world(self):
        self.seek_start = self.random_pos(self.shape[0]//2, self.shape[1]//2)
        self.hider_start = self.random_pos()
