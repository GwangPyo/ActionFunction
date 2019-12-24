import envWrapper
import numpy as np
import tensorflow as tf
from stable_baselines.common.distributions import make_proba_dist_type
from stable_baselines.common import tf_util
import gym
from replayBuf import replaybuffer


class FunctionEnv(envWrapper.WrappedEnvClass):
    """
    Wrapping Environment for function Action
    """
    DISCRETE = 0
    CONTINUOUS = 1

    def __init__(self, wrappedEnv, num_seq, neuro_structure, ):
        """
        :param wrappedEnv: The target environment object that shall be wrapped
        :param num_seq: The life time of the action function. Maybe variable length
        :param neuro_structure: The structure of action function. data type is tuple.
        e.g.,
        if local observation space shape is 3, and neuro_structure is (3,4,5)
        Then it will construct
        (3, 3), (3, 4), (4, 5)
        shape of neural network
        """
        super().__init__(wrappedEnv, num_seq)
        assert type(neuro_structure) is tuple

        # check wrapped environment's action space type
        if type(self.wrapped_env.action_space) is gym.spaces.Box:
            self.mode = FunctionEnv.DISCRETE
        else:
            self.mode = FunctionEnv.CONTINUOUS

        # tensorflow session for the neural network
        self.sess = tf.Session()
        # observation space
        obs_shape = list(self.wrapped_env.observation_space.shape)
        obs_shape.insert(0, None)
        self.obs = tf.placeholder(tf.float32, shape=obs_shape)
        a = 1
        for sh in range(len(self.wrapped_env.observation_space.shape)):
            a *= sh
        for sh in neuro_structure:
            a *= sh
        self.action_space = gym.spaces.Box(low=-3, high=3, shape=(a, ))
        self._pdtype = make_proba_dist_type(self.action_space)
        self._proba_distribution = None
        self.action_ph = None
        self._policy_proba = None
        self.pg_loss = None
        self.params = None
        self.neuro_structure = neuro_structure
        self.policy = self.init_network('net')
        self.neuro_structure = self.parse_neuro_structure(neuro_structure)
        self.last_state = None
        self.step_cnt = 0
        self.replay_buffer = replaybuffer(maxlen=512)
        self.sess.run(tf.global_variables_initializer())

    def reset(self):
        self.last_state = super().reset()
        return np.copy(self.last_state)

    def step(self, action):
        action = np.sinh(action)
        temp_neuron = self.build_neurons(action)
        r_cnt = 0
        done = False
        info = {}
        self.step_cnt += 1
        seq = self.num_seq
        for _ in range(seq):
            a = self.run_neuro_discrete(self.last_state, temp_neuron)
            s, r, done, info = self.wrapped_env.step(a)
            self.replay_buffer.add({'state': s, 'reward': r, 'done': done, 'action': a})
            self.last_state = s
            r_cnt += r
            if done:
                return s, r_cnt, done, info
        return self.last_state, r_cnt, done , info

    @staticmethod
    def run_neuro_discrete(state, neurons):
        x = np.copy(state)
        for n in range(len(neurons) - 1):
            layer = neurons[n]
            x = np.matmul(x, layer)
            x = np.maximum(x, 0, x)
        x = np.matmul(x, neurons[-1])
        x = np.tanh(x)
        return np.argmax(x)

    def init_network(self, name):
        """
        Initialize networks
        :param name: variable scope names
        :return: temporal policy.
        """
        with tf.variable_scope(name):
            # build temporal neural networks
            if len(self.neuro_structure) > 2:
                model = tf.layers.dense(inputs=self.obs, units=self.neuro_structure[0], trainable=False)

                for i in range(len(self.neuro_structure) -1):
                    model = tf.layers.dense(model, self.neuro_structure[i], activation=tf.nn.relu)
                if self.mode == FunctionEnv.DISCRETE:
                    model = tf.layers.dense(model, self.neuro_structure[-1], activation=tf.nn.softmax)
                else:
                    model = tf.layers.dense(model, self.neuro_structure[-1], activation=tf.nn.sigmoid)
            else:
                if self.mode == FunctionEnv.DISCRETE:
                    model = tf.layers.dense(inputs=self.obs,
                                            units=self.neuro_structure[0], trainable=False, activation=tf.nn.softmax)
                else:
                    model = tf.layers.dense(inputs=self.obs,
                                            units=self.neuro_structure[0], trainable=False, activation=tf.nn.softmax)
        # build probability distribution (value function and advantage)

        self._proba_distribution, _, _ = \
            self._pdtype.proba_distribution_from_latent(model, model, init_scale=0.01)

        # probability distribution of action
        self.action_ph = self._pdtype.sample_placeholder([None], name='action_ph')
        self._policy_proba = [self._proba_distribution.mean, self._proba_distribution.std]
        self.params = tf_util.get_trainable_vars(name)
        self.pg_loss = tf.gradients(self._proba_distribution.neglogp(self.action_ph), self.params)
        return model

    def predict(self, observation):

        action = self.sess.run([self.policy], {self.obs: observation})
        if self.mode == FunctionEnv.DISCRETE:
            return np.random.choice(a=np.arange(len(action)),  p=action)
        elif self.mode == FunctionEnv.CONTINUOUS:
            return action
        else:
            raise KeyError

    def get_gradeints(self):
        gradient_set = []
        dataset = self.replay_buffer.get_data()
        for replay in dataset:
            G = replay['reward']
            gradient = self.sess.run([self.pg_loss], feed_dict={self.obs: replay['state'], self.action_ph: replay['action']})
            gradient_set.append(gradient * G)
        return gradient_set

    def parse_neuro_structure(self, action_structure) -> list:
        assert len(action_structure) >= 2
        if len(action_structure) == 2:
            return [action_structure]
        else:
            shapes = []
            for i in range(len(action_structure) - 1):
                shapes.append((action_structure[i], action_structure[i + 1]))
            return shapes

    def build_neurons(self, action):
        neurons = []
        for i in range(len(self.partition_table) - 1):
            a = action[self.partition_table[i]: self.partition_table[i + 1]]
            a = a.reshape(self.neuro_structure[i])
            neurons.append(a)
        return neurons


