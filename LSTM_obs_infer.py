import keras
import numpy as np


def seq2dataset(seq, window_size):
    seq_length = seq.shape[0] - 1
    shape = seq.shape[1: ]
    data_shape = (seq_length, ) + (window_size, ) + shape
    data_set = np.zeros(shape=data_shape)
    for i in range(seq_length):
        sub_seq = seq[i - i % window_size: i + 1]
        data_set[i, 0: i % window_size + 1, :] = sub_seq
    return data_set


class ObservationInference(object):
    def __init__(self, obs_shape, action_shape, time_step=5):
        self.obs_shape = obs_shape
        self.obs_size = 1
        for i in self.obs_shape:
            self.obs_size *= i
        self.time_step = time_step
        self.action_shape= action_shape
        self.model = self._build_model()
        self.model.compile(optimizer="Adam", loss="mse")

    def _build_model(self):
        obs_input = keras.layers.Input(shape=self.obs_shape)
        embedding = keras.layers.Dense(64, activation='tanh')(obs_input)
        action_input = keras.layers.Input(shape=(None, ) + self.action_shape)
        action_embedding = keras.layers.Dense(64, activation='tanh')(action_input)
        action_lstm = keras.layers.LSTM(units=8, return_sequences=True)(action_embedding)
        action_lstm = keras.layers.LSTM(units=8, return_sequences=True)(action_lstm)
        action_lstm = keras.layers.LSTM(units=8)(action_lstm)
        action_embedded = keras.layers.Dense(64)(action_lstm)
        merge = keras.layers.Concatenate()([embedding, action_embedded])
        embedding = keras.layers.Dense(64, activation='relu')(merge)
        inferred_obs = keras.layers.Dense(self.obs_size)(embedding)
        model = keras.models.Model(input=[obs_input, action_input], output=inferred_obs)
        return model

    def train(self, path="CartPoleTrajectory.npz"):
        data = np.load(path)
        actions = data["actions"]
        actions = keras.utils.np_utils.to_categorical(actions)
        actions = seq2dataset(actions, self.time_step)
        obs = data["obs"]
        obs_x = []
        for i in range(len(obs)):
            if i % self.time_step == 0:
                obs_x.append(obs[i])
            else:
                obs_x.append(obs[i - (i % self.time_step)])
        obs_x = obs_x[: -1]
        obs_x = np.array(obs_x)
        obs_y = obs[1: ]
        self.model.fit(x=[obs_x, actions], y=obs_y, epochs=5, batch_size=64)

    def save_weights(self):
        self.model.save_weights("lstm_weights.h5")

    def load_weights(self):
        self.model.load_weights("lstm_weights.h5")


if __name__ == "__main__":
    TIME_STEP = 15
    inferrence_model = ObservationInference(obs_shape=(4, ), action_shape=(2, ), time_step=TIME_STEP)
    inferrence_model.train()
    inferrence_model.save_weights()
    inferrence_model.load_weights()
    valid_data = np.load("CartPoleTrajectoryValidation.npz")
    obs = valid_data["obs"]

    valid_action = valid_data["actions"]
    valid_action = keras.utils.np_utils.to_categorical(valid_action)
    valid_action = seq2dataset(valid_action, TIME_STEP)
    avg_mse = []
    for i in range(len(obs) - TIME_STEP):
        if i >= 5:
            actions = valid_action[i]
            actions = np.expand_dims(actions, axis=0)
            obs_x = obs[i - i % TIME_STEP]
            obs_x = np.expand_dims(obs_x, axis=0)
            print("actual")
            print(obs[i + 1].flatten())
            prediction = inferrence_model.model.predict_on_batch([obs_x, actions])
            print("prediction")
            print(prediction.flatten())
            mse =  np.sum(np.square(obs[i + 1].flatten() - prediction.flatten()))
            print("mse", np.mean(np.square(obs[i + 1].flatten() - prediction.flatten())))
            avg_mse.append(mse)

    print(np.mean(avg_mse))