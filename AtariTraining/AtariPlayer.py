from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common import make_vec_env
import gym
import numpy as np

if __name__ == "__main__":
    env = make_vec_env(env_id="Pong-v0", n_envs=12)
    model = PPO2(env=env, policy=CnnPolicy, verbose=1, n_steps=512, tensorboard_log="./")
    model.learn(50000000)
    model.save("PPOModel.pkl")
    env = gym.make("Breakout-v0")

    obs_capture = []
    while len(obs_capture) < 10000:
        done = False
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            obs_capture.append(obs)
    obs_capture = np.array(obs_capture, dtype=np.float32)
    np.save("obs_capture", obs_capture)
