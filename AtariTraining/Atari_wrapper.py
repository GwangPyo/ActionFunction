import gym
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.io import imsave
import pickle as pkl
from GAN import AAE


class AtariWrapEnv(gym.Env):
    def __init__(self, game_id, frame_skip = 4, render_mode=False, sampling_mode=False):
        self.game_id = game_id
        self.atari = gym.make(game_id, obs_type='image', frameskip=1)
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.real_state = []
        self.action_space = self.atari.action_space
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, self.frame_skip))
        self.num_timesteps = 0
        self.sum_reward = 0

        self.sampling_mode = sampling_mode
        self.sampling_number = 0
        self.sampling_data = []

    def step(self, action):
        done = False
        reward = 0
        self.real_state = []

        for _ in range(self.frame_skip):
            if done:
                self.real_state.append(np.zeros((84, 84)))
                continue
            state, temp_reward, done, _ = self.atari.step(action)
            state = resize(rgb2gray(state), (84, 84))
            if done:
                saved_sample = np.uint8(state * 255)
                imsave('image_sample.png', saved_sample)
            self.real_state.append(state)
            reward += temp_reward
            if self.render_mode:
                self.render()
        self.real_state = np.array(self.real_state)
        self.sum_reward += reward
        self.num_timesteps += 1
        output = self.real_state

        if self.sampling_mode:
            self.sampling_data.append(output)
            if len(self.sampling_data) == 100000:
                with open('./sampling_data/samples_{}.pkl'.format(self.sampling_number), 'wb') as f:
                    pkl.dump(self.sampling_data, f)
                    self.sampling_number += 1

        if done:
            info = {'episode': {'r': self.sum_reward, 'l': self.num_timesteps}, 'game_reward': reward}
        else:
            info = {'episode': None, 'game_reward': reward}

        return output.transpose(), reward, done, info

    def reset(self):
        self.real_state = []
        for i in range(self.frame_skip - 1):
            self.real_state.append(np.zeros((84, 84)))
        state = self.atari.reset()
        state = resize(rgb2gray(state), (84, 84))
        self.real_state.append(state)
        # imshow(state)

        self.sum_reward = 0
        self.num_timesteps = 0
        return np.array(self.real_state).transpose()

    def render(self, mode="HUMAN"):
        self.atari.render()


class AtariWrapStateEmbedding(gym.Env):
    def __init__(self, game_id, frame_skip=4, render_mode=False):
        self.game_id = game_id
        self.atari = gym.make(game_id, obs_type='image', frameskip=1)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.real_state = []
        self.action_space = self.atari.action_space

        aae = AAE(shape=(20, 80, 4))
        self.embedding = aae.encoder
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(aae.latent_dim, ))

    @property
    def state(self):
        s = np.array(self.real_state)
        s = np.transpose(s, axes=[1, 2, 0])
        s = s[-20: , 2:-2, :]
        s = np.expand_dims(s, axis=0)
        s = self.embedding.predict_on_batch(s).flatten()
        return s

    def reset(self):
        self.real_state = []
        for i in range(self.frame_skip - 1):
            self.real_state.append(np.zeros((84, 84)))
        state = self.atari.reset()
        state = resize(rgb2gray(state), (84, 84))
        self.real_state.append(state)
        self.sum_reward = 0
        self.num_timesteps = 0
        return self.state

    def step(self, action):
        done = False
        reward = 0
        self.real_state = []

        for _ in range(self.frame_skip):
            if done:
                self.real_state.append(np.zeros((84, 84)))
                continue
            state, temp_reward, done, _ = self.atari.step(action)
            state = resize(rgb2gray(state), (84, 84))
            self.real_state.append(state)
            reward += temp_reward
            if self.render_mode:
                self.render()

        self.sum_reward += reward
        self.num_timesteps += 1
        if done:
            info = {'episode': {'r': self.sum_reward, 'l': self.num_timesteps}, 'game_reward': reward}
        else:
            info = {'episode': None, 'game_reward': reward}

        return self.state, reward, done, info

    def render(self, mode="HUMAN"):
        self.atari.render()



from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    env = SubprocVecEnv([lambda:AtariWrapStateEmbedding("Breakout-v0") for _ in range(8)])
    model = PPO2(MlpPolicy, env)
    model.learn(10000)
    model.save("BreakoutPPO2_embedding.pkl")
