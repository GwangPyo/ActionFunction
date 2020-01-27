import gym
from stable_baselines import SAC
from stable_baselines.sac.policies import LnMlpPolicy
from stable_baselines.common import make_vec_env

if __name__ == "__main__":
    env = gym.make("BipedalWalkerHardcore-v2")
    model = SAC(env=env, policy=LnMlpPolicy, verbose=1)
    model.learn(int(1e+6))
    model.save("Bipedal-SAC.pkl")