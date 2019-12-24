from  FunctionWrappedEnv import FunctionEnv
import gym
from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy
import numpy as np


if __name__ == "__main__":
    cartpole = gym.make("CartPole-v1")
    print(cartpole.observation_space)
    neuro_structure=  (2, )
    env = FunctionEnv(cartpole, 20, neuro_structure)

    model = SAC(LnMlpPolicy, env, verbose=0, full_tensorboard_log=True,
                tensorboard_log="D:\논문\ActionLearningTensorBoard")
    model.learn(1000000)

    obs_list = []
    action_list = []
    obs = env.reset()
    for _ in range(1000):
        obs_list.append(np.copy(obs))
        action, _ = model.predict(obs)
        action_list.append(np.copy(action))
        obs, r, d, i = env.step(action)
        if d:
            obs = env.reset()

    np.save("obs", np.array(obs_list))
    np.save("action", np.array(action_list))