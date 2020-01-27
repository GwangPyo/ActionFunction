from skimage.color import rgb2gray
from skimage.transform import resize
import os,sys,inspect
import gym
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import FunctionWrappedEnv
from encoding import AAE


class AtariWrapStateEmbedding(FunctionWrappedEnv.FunctionEnv):
    def __init__(self, game_id, num_seq, neuro_structure, frame_skip=4, render_mode=False):
        self.game_id = game_id
        self.atari = gym.make(game_id, obs_type='image', frameskip=1)
        super().__init__(self.atari, num_seq, neuro_structure)
        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.real_state = []
        aae = AAE(shape=(20, 80, 4))
        self.embedding = aae.encoder
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(84, 84, self.frame_skip))
        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(self.partition_table[-1], ))

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
        self.last_state = np.copy(self.state)
        return np.transpose(np.array(self.real_state), axes=[1, 2, 0])

    def step(self, action):
        done = False
        reward = 0
        action = np.sinh(action)
        temp_neuron = self.build_neurons(action)
        state = None
        for i in range(self.num_seq):
            a = self.run_neuro_discrete(self.last_state, temp_neuron)
            self.real_state = []
            for _ in range(self.frame_skip):
                if done:
                    self.real_state.append(np.zeros((84, 84)))
                    continue
                state, temp_reward, done, _ = self.atari.step(a)
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

        return np.transpose(np.array(self.real_state), axes=[1, 2, 0]) , reward, done, info

    def render(self, mode="HUMAN"):
        self.atari.render()

    def build_action_partion_table(self):
        cnt = 0
        table = [0]
        sh_1 = self.inner_model_shape[0][1]
        self.inner_model_shape[0] = (8, sh_1)
        for sh in self.inner_model_shape:
            size = sh[0] * sh[1]
            cnt += size
            table.append(cnt)
        return table

    def init_network(self,  name):
        """
        Initialize networks
        :param name: variable scope names
        :return: temporal policy.
        """
        shapes = []
        list_shape = self.obs.shape.as_list()
        if list_shape[0] is None:
            del list_shape[0]
        if len(list_shape) == 1:
            list_shape = list_shape[0]
        else:
            list_shape = tuple(list_shape)

        # build temporal neural networks
        if len(self.neuro_structure) >= 2:
            shapes.append((list_shape, self.neuro_structure[0]))
            for i in range(len(self.neuro_structure) -1):
                shapes.append((self.neuro_structure[i], self.neuro_structure[i + 1]))
        else:
            shapes.append((list_shape, self.neuro_structure[0]))

        # inner model shape is a variable which denotes the shapes of the temporal neural networks
        # for instance (2, 4) with observation space (4, ) the variable will be
        # [ (4, 2), (2, 4)]
        # this is structure of the neural networks
        self.inner_model_shape = shapes
        return



from stable_baselines import PPO2
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv


if __name__ == "__main__":
    env = SubprocVecEnv([lambda: AtariWrapStateEmbedding("Breakout-v0", num_seq=5, neuro_structure=(4, 4)) for _ in range(8)])
    model = PPO2(CnnPolicy, env, verbose=1)
    model.learn(10000000)
    model.save("BreakoutPPO2_embedding.pkl")
