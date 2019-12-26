from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
# from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
import numpy as np
import gym


if __name__ == '__main__':
    env_test = FunctionWrappedEnv()