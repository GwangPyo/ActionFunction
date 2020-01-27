import gym
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from keras import layers as layers
from keras.models import Model
import numpy as np


def rgb2gray(rgb):
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=-1)


def obs_type(gray_scale):
    if gray_scale:
        def converted_obs(obs):
            return rgb2gray(obs)
    else:
        def converted_obs(obs):
            return obs
    return converted_obs


def crop(obs, mario_pos):
    # clip mario position
    if mario_pos[0] < 30:
        mario_pos[0] = 30

    elif mario_pos[0] > 210:
        mario_pos[0] = 210

    if mario_pos[1] < 32:
        mario_pos[1] = 32

    elif mario_pos[1] > 224:
        mario_pos[1] = 224

    return obs[mario_pos[0] - 30: mario_pos[0] + 30, mario_pos[1] - 32: mario_pos[1] + 32, ...]


class MarioEnvWrapper(gym.Env):
    FULL_OBS_COLOR = (240, 256, 3)
    FULL_OBS_GRAY = (240, 256, 1)
    PARTIAL_OBS_COLOR = (60, 64, 3)
    PARTIAL_OBS_GRAY = (60, 64, 1)

    STATUS_MAP = {"small": 0, 'tall':1, 'fireball':2}

    def __init__(self, gray_scale=True, intrinsic=False, full_obs=True):
        self.env = JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), SIMPLE_MOVEMENT)
        self.gray_scale = gray_scale
        self.full_obs = full_obs
        self.convert_obs = obs_type(gray_scale)

        self.current_coin = 0
        self.current_status = 0
        self.current_score = 0
        self.intrinsic = intrinsic
        if intrinsic:
            self.intrinsic_target = self._build_intrinsic()
            self.intrinsic_layer = self._build_intrinsic()
            self.intrinsic_target.compile(loss="mse", optimizer="sgd")
            self.intrinsic_layer.compile(loss="mse", optimizer="sgd")
        else:
            self.intrinsic_target = None
            self.intrinsic_layer = None

    def render(self, mode='human'):
        return self.env.render(mode=mode)

    def reset(self):
        self.current_coin = 0
        self.current_status = 0
        self.current_score = 0
        if self.full_obs:
            return self.convert_obs(self.env.reset())
        else:
            return crop(obs=self.convert_obs(self.env.reset()), mario_pos=[40, 80])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.convert_obs(obs)

        if not self.full_obs:
            mario_pos = [info["x_pos_screen"], info["y_pos"]]
            obs =crop(obs, mario_pos)

        if self.intrinsic:
            _obs = np.expand_dims(obs, axis=0)
            y = self.intrinsic_target.predict(_obs)
            intrinsic_reward = self.intrinsic_layer.train_on_batch(x=_obs, y=y)
            reward = intrinsic_reward

        return obs, reward, done, info

    def _build_intrinsic(self):
        obs = layers.Input(shape=self.observation_space.shape)
        x = layers.Conv2D(kernel_size=(9, 9), strides=3, filters=24, padding="SAME", kernel_initializer="Orthogonal")(obs)
        x = layers.Conv2D(kernel_size=(5, 5), strides=3, filters=12, padding="SAME", kernel_initializer="Orthogonal")(x)
        x = layers.Conv2D(kernel_size=(5, 5), strides=3, filters=1, padding="SAME", kernel_initializer="Orthogonal")(x)
        f = layers.Flatten()(x)
        y = layers.Dense(12, kernel_initializer="glorot_uniform", activation="sigmoid")(f)
        mode = Model(obs, y)
        mode.summary()
        return mode

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        if self.full_obs:
            if self.gray_scale:
                return gym.spaces.Box(low=0, high=255, shape=MarioEnvWrapper.FULL_OBS_GRAY)
            else:
                return self.env.observation_space
        else:
            if self.gray_scale:
                return gym.spaces.Box(low=0, high=255, shape=MarioEnvWrapper.PARTIAL_OBS_GRAY)
            else:
                return gym.spaces.Box(low=0, high=255, shape=MarioEnvWrapper.PARTIAL_OBS_COLOR)

    def coin(self, info):
        if info["coins"] > self.current_coin:
            self.current_coin = info["coins"]
            return info["coins"] - self.current_coin
        else:
            return 0

    def mario_status(self, info):
        if MarioEnvWrapper.STATUS_MAP[info["status"]] > self.current_status:
            self.current_status = MarioEnvWrapper.STATUS_MAP[info["status"]]
            return 5
        elif MarioEnvWrapper.STATUS_MAP[info["status"]]< self.current_status:
            self.current_status = MarioEnvWrapper.STATUS_MAP[info["status"]]
            return -5
        else:
            return 0

    def score(self, info):
        return (info["score"] - self.current_score)/100

    def final(self, done, info):
        flag_get = info["flag_get"]
        if flag_get and done:
            return 10
        elif done and not flag_get:
            return -10
        else:
            return 0

    def info_reward(self, done, info):
        reward = 0
        reward += self.coin(info)
        reward += self.mario_status(info)
        reward += self.final(done, info)
        return reward


class AggressiveMarioEnv(MarioEnvWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.convert_obs(obs)
        if not self.full_obs:
            mario_pos = [info["x_pos_screen"], info["y_pos"]]
            obs =crop(obs, mario_pos)

        if self.intrinsic:
            _obs = np.expand_dims(obs, axis=0)
            y = self.intrinsic_target.predict(_obs)
            intrinsic_reward = self.intrinsic_layer.train_on_batch(x=_obs, y=y)
            reward = intrinsic_reward
        else:
            reward = 0
        reward += self.info_reward(done, info)
        




from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env


if __name__ == "__main__":
    env_lambda = lambda: AggressiveMarioEnv(intrinsic=True, full_obs=False)
    env = make_vec_env(env_lambda, n_envs=6)
    model = PPO2(policy=CnnPolicy, env=env, verbose=1)
    model.learn(100000)
    model.save("supermario_partial_aggressive.pkl")
    done = False
    env = env_lambda()
    obs = env.reset()
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
