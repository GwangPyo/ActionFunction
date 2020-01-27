import gym
from stable_baselines.deepq.replay_buffer import ReplayBuffer
import numpy as np
import pickle
import keras
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines import DQN, PPO2, SAC
from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.gail import generate_expert_traj
from keras.utils import np_utils
from LSTM_obs_infer import ObservationInference
from collections import deque


class CartPoleObservationInference(gym.Env):
    def __init__(self, freq=5):
        self.cartpole = gym.make("CartPole-v1")
        self.observation_space = self.cartpole.observation_space
        self.action_space = self.cartpole.action_space
        self.last_obs = None
        self.action_seq = []
        self.buffer = ReplayBuffer(size=50000)
        self.observation_model = ObservationInference(obs_shape=(4, ), action_shape=(2, ), time_step=freq)
        self.observation_model.train()
        # self.observation_model.load_weights()
        self.step_cnt = 0
        self.freq = freq
        self.action_batch = None
        self.last_mse = None
        self.total_mse = None

    def reset(self):
        self.last_obs = self.cartpole.reset()
        self.step_cnt = 0
        self.action_seq = []
        self.action_batch = np.zeros(shape=(1, self.freq, 2))
        self.last_mse = deque(iterable=np.zeros(self.freq), maxlen=self.freq)
        self.total_mse = []
        return self.last_obs

    def render(self, mode='human'):
        return self.cartpole.render(mode)

    def step(self, action):
        self.action_seq.append(action)
        obs_real, reward, done, info = self.cartpole.step(action)
        obs_batch = np.expand_dims(self.last_obs, axis=0)
        self.action_batch[0, self.step_cnt] = np_utils.to_categorical(action, num_classes=2)
        self.step_cnt += 1
        self.step_cnt = self.step_cnt % self.freq
        if self.step_cnt != 0:
            obs = self.observation_model.model.predict_on_batch([obs_batch, self.action_batch])

            obs = obs.flatten()
            info["mse"] = np.mean(np.square(np.abs(obs - obs_real)))

        else:
            obs = obs_real
            self.action_batch = np.zeros(shape=(1, self.freq, 2))
            self.last_obs = np.copy(obs_real)
            self.action_seq = []
            info["mse"] = 0

        self.last_mse.append(info["mse"])
        self.total_mse.append(info["mse"])
        return obs, reward, done, info



class CartPoleStateInfer(gym.Env):
    def __init__(self, model, time_step=5):
        self.cartpole = gym.make("CartPole-v1")
        obs_shape = self.cartpole.observation_space.shape[0] + 2
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape, ))
        self.action_space = gym.spaces.Box(low=-20, high= 20, shape=self.cartpole.observation_space.shape)
        self.model = model
        self.obs_real = None
        self.last_action = np.zeros(2)
        self.time_step = time_step


    def reset(self):
        obs_cartpole = self.cartpole.reset()
        self.obs_real = obs_cartpole
        obs_action = np.zeros(2)
        return np.concatenate([obs_cartpole, obs_action])

    def step(self, action):
        model_action, _  = self.model.predict(action)
        obs_real, reward, done, info  = self.cartpole.step(model_action)
        obs_action = np.zeros(2)
        obs_action[model_action] = 1
        obs = np.concatenate([obs_real, obs_action])
        return obs, reward, done, info

    def render(self, mode='human'):
        self.cartpole.render(mode='human')

import pandas as pd
if __name__ == "__main__":
    model = PPO2.load("CartPole-PPO2.pkl")
    infer_steps = [5, 10, 15]
    for i in infer_steps:
        env = CartPoleObservationInference(i)
        data_frame = pd.DataFrame(columns=["score", "last_10_mse", "total_mse"])
        for e in range(100):
            done = False
            score = 0
            obs = env.reset()
            while not done:
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                score += reward
            data_frame.loc[-1] = ([score, np.mean(env.last_mse), np.mean(env.total_mse)])
            data_frame.index += 1

            print(data_frame.tail(1))
        data_frame = data_frame.sort_values(by=['score'])
        data_frame.to_csv("inference_model_score_{}.csv".format(i))
