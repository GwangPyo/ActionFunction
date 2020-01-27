from  FunctionWrappedEnv import FunctionEnv
import gym
import numpy as np
from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy


from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    lunarLander = gym.make("LunarLander-v2")

    print(lunarLander.observation_space)
    # env = SubprocVecEnv([lambda : lunarLander for _ in range(8)])

    neuro_structure=  (8, 4, 4)
    env = SubprocVecEnv([lambda: FunctionEnv(lunarLander, (20), neuro_structure) for _ in range(8)])

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./")
    model.learn(50000000)
    model.save("D:/ActionFunctionData/15/15_model_SAC.pkl")

    env = FunctionEnv(lunarLander, (20), neuro_structure)
    reward_list = []
    for _ in range(100):
        done = False
        reward_cnt = 0
        obs = env.reset()
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            reward_cnt += reward
        reward_list.append(reward_cnt)
    print(np.mean(reward_list))
    """
    model = PPO2(MlpPolicy, env=env, verbose=1)
    model.learn(50000000)

    env = lunarLander
  
    """