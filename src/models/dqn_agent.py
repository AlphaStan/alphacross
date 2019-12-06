import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten

from src.models.agent import _Agent


#TODO: test class for that
class DQNAgent(_Agent):

    def __init__(self,
                 state_space_size,
                 action_space_size,
                 env,
                 epsilon=0.05,
                 discount=0.95,
                 num_episodes=1000,
                 mini_batch_size=200,
                 num_replay=1000
                 ):
        super().__init__()
        self.model = self.init_model(state_space_size, action_space_size)
        self.epsilon = epsilon
        self.discount = discount
        self.num_episodes = num_episodes
        self.mini_batch_size = mini_batch_size
        self.num_replay = num_replay
        self.replays = self.init_replays(env)

    @staticmethod
    def init_model(state_space_size, action_space_size):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=(7, 6)))
        model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu, input_dim=state_space_size))
        #model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(action_space_size, activation=tf.keras.activations.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        print(model.summary())
        return model

    def init_replays(self, env):
        replays = []
        while len(replays) < self.num_replay:
            replays += self.generate_replay(env)
        return replays[:self.num_replay]

    def generate_replay(self, env):
        replays = []
        game_is_not_finished = True
        while game_is_not_finished:
            prior_state = env.get_state().reshape((1, 7, 6))
            actions = self.model.predict(prior_state)
            action = self.epsilon_greedy_predict_action(actions)
            reward, new_state = env.apply_action(action)
            replay = Replay(prior_state, action, reward, new_state)
            replays.append(replay)

            game_is_not_finished = not env.is_terminal_state(env.get_state())
        return replays

    def play_action(self, env):
        state = env.get_state().reshape((1, 7, 6))
        action_probabilities = self.model.predict(state)
        action_id = self.epsilon_greedy_predict_action(action_probabilities)
        reward, new_state = env.apply_action(action_id)
        return reward, new_state

    def train(self, env):
        for episode in range(self.num_episodes):
            game_is_not_finished = True
            while game_is_not_finished:
                prior_state = env.get_state().reshape((1, 7, 6))
                actions = self.model.predict(prior_state)
                action = self.epsilon_greedy_predict_action(actions)
                reward, new_state = env.apply_action(action)
                replay = Replay(prior_state, action, reward, new_state)
                self.save_replay(replay)
                mini_batch = self.sample_minibatch()
                mini_batch_targets = self.get_mini_batch_targets(mini_batch)
                mini_batch_states = [replay.post_state for replay in mini_batch]
                self.model.train_on_batch(mini_batch_states, mini_batch_targets)

                game_is_not_finished = not env.is_terminal_state(env.get_state())

    def save_replay(self, replay):
        self.replays.pop(0)
        self.replays.append(replay)

    def sample_minibatch(self):
        return np.random.choice(self.replays, self.mini_batch_size, replace=False)

    def get_mini_batch_targets(self, mini_batch):
        return [replay._reward if replay._reward == 1
            else self.discount * np.max(self.model.predict(replay._post_state)) for replay in mini_batch]

    def epsilon_greedy_predict_action(self, actions):
        if np.random.random_sample() < self.epsilon:
            return np.random.randint(0, len(actions))
        else:
            return np.argmax(actions)


class Replay:
    def __init__(self, prior_state, action, reward, post_state):
        self._prior_state = prior_state
        self._action = action
        self._reward = reward
        self._post_state = post_state
