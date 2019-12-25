import gym
from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    env = SubprocVecEnv([lambda: gym.make("Breakout-v0") for _ in range(16)])
    model = PPO2(CnnPolicy, env, verbose=1, n_steps=256)
    model.learn(10000000)
    model.save("PPO_breakout.pkl")