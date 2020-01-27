import gym
import ray
import ray.rllib.agents.ars as ars
import numpy as np
from ray.tune.logger import pretty_print
import pickle
import tensorflow as tf


class BipedalWalkerWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make("BipedalWalker-v2")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self):
        ret = self.env.reset()
        return ret

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human'):
        return self.env.render(mode)


if __name__ == "__main__":
    ray.init()
    config = ars.DEFAULT_CONFIG.copy()
    config["num_workers"] = 1
    config["eager"] = False
    trainer = ars.ARSTrainer(config=config, env="BipedalWalkerHardcore-v2")
    sess = trainer.sess
    writer = tf.summary.FileWriter("ARS_GRAPH")
    writer.add_graph(sess.graph)
    writer.close()
    """
    trainer.restore("/home/yoo/ray_results/ARS_BipedalWalkerHardcore-v2_2020-01-13_18-57-58f6o8cmvr/checkpoint_901/checkpoint-901")
    weight = trainer.policy.variables
    
    print(weight.get_weights())
    """