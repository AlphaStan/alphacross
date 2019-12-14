import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten
from ..main.errors import ColumnIsFullError
from src.models.agent import _Agent


#TODO: test class for that
class DQNAgent(_Agent):

    def __init__(self,
                 state_space_size,
                 action_space_size,
                 env,
                 epsilon=0.25,
                 discount=0.95,
                 num_episodes=1000,
                 mini_batch_size=200,
                 num_replay=1000
                 ):
        super().__init__()
        self.action_space_size = action_space_size
        self.model = self.init_model(state_space_size, action_space_size)
        self.epsilon = epsilon
        self.discount = discount
        self.num_episodes = num_episodes
        self.mini_batch_size = mini_batch_size
        self.num_replay = num_replay
        self.replays = []

    @staticmethod
    def init_model(state_space_size, action_space_size):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=(7, 6)))
        model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu, input_dim=state_space_size))
        model.add(tf.keras.layers.Dense(action_space_size, activation=tf.keras.activations.softmax))
        model.compile(loss=tf.losses.huber_loss, optimizer='Adam', metrics=['accuracy'])
        return model

    def init_replays(self, env):
        replays = []
        while len(replays) < self.num_replay:
            replays += self.generate_replay(env)
            env.reset()
            print("Replays generated %i/%i" % (min(len(replays), self.num_replay), self.num_replay))
        self.replays = replays[:self.num_replay]

    def generate_replay(self, env):
        replays = []
        game_is_finished = False
        while not game_is_finished:
            prior_state = np.expand_dims(env.get_np_array().reshape((7, 6)), axis=0)
            action = np.random.choice(self.action_space_size)
            try:
                reward, new_state = env.apply_action(action)
            except ColumnIsFullError:
                continue
            replay = Replay(prior_state, action, reward, new_state)
            replays.append(replay)
            game_is_finished = env.is_terminal_state(env.get_state())
            if env.is_blocked():
                game_is_finished = True
        return replays

    def play_action(self, env):
        state = env.get_np_array().reshape((1, 7, 6))
        action_probabilities = self.model.predict(state)
        action_id = self.epsilon_greedy_predict_action(action_probabilities)
        reward, new_state = env.apply_action(action_id)
        return reward, new_state

    def train(self, env):
        if not self.replays:
            raise AttributeError("Attempting to train DQNAgent with no replays, use generate_replays first")
        for episode in range(self.num_episodes):
            print("Train on episode %i/%i" % (episode, self.num_episodes))
            game_is_finished = False
            while not game_is_finished:
                prior_state = env.get_np_array().reshape((1, 7, 6))
                actions = self.model.predict(prior_state).ravel()
                action = self.epsilon_greedy_predict_action(actions)
                try:
                    reward, new_state = env.apply_action(action)
                    replay = Replay(prior_state, action, reward, new_state)
                    self.save_replay(replay)
                    mini_batch = self.sample_minibatch()
                    mini_batch_targets = self.get_mini_batch_targets(mini_batch)
                    mini_batch_states = np.array([replay._post_state for replay in mini_batch])
                    #TODO: Fix wrong-sized targets
                    self.model.train_on_batch(mini_batch_states, mini_batch_targets)
                    game_is_finished = env.is_terminal_state(env.get_state())
                except(ColumnIsFullError):
                    continue
            env.reset()


    def get_legal_action(self, state, get_action_prob, env):
        action_probabilities = get_action_prob(state)
        legal_actions = [env.is_legal_action(state, action) for action in range(action_probabilities)]
        legal_action_prob = [prob if legal_action else 0
                             for prob, legal_action in zip(action_probabilities, legal_actions)]

        if sum(legal_action_prob) == 0:
            return legal_action_prob
        return legal_action_prob / sum(legal_action_prob)


    def save_replay(self, replay):
        self.replays.pop(0)
        self.replays.append(replay)

    def sample_minibatch(self):
        return np.random.choice(self.replays, self.mini_batch_size, replace=False)

    def get_mini_batch_targets(self, mini_batch):
        return np.array([replay._reward if replay._reward == 1
                         else self.discount * np.max(self.model.predict(np.expand_dims(replay._post_state, axis=0)))
                         for replay in mini_batch])

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
