import numpy as np
import tensorflow as tf
from src.main.cross_game import CrossGame

epsilon = 0.05

c = CrossGame()
nb_column = c.nb_columns




def train_model():
    gameIsNotFinished = True
    counter = 0
    c = CrossGame()
    while(gameIsNotFinished):
        action = model.epsilon_greedy_predict_action()
        reward = c.play_token(action, counter%2)
        counter += 1

class Agent:

    def __init__(self, num_episodes = 1000, mini_batch_size = 200):
        self.model = self.init_model()
        self.replays = self.init_replays()
        self.num_episodes = num_episodes
        self.mini_batch_size = mini_batch_size

    @staticmethod
    def init_model():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(c.nb_rows * c.nb_columns, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(c.nb_columns, activation=tf.keras.activations.softmax))
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        return model

    def init_replays(self):
        replays = []
        while len(replays) < 1000:
            replays += self.generate_replay()
        return replays

    def generate_replay(self):
        return

    def play_action(self, env):
        state = env.get_state()
        action_probabilities = self.model.predict(state)
        action_id = self.epsilon_greedy_predict_action(action_probabilities)
        reward, new_state = env.apply_action(action_id)
        return reward, new_state

    def train_action(self, env):
        for episode in range(self.num_episodes):
            gameIsNotFinished = False
            while gameIsNotFinished:
                prior_state = env.get_state()
                actions = self.model.predict(prior_state)
                action = self.epsilon_greedy_predict_action(actions)
                reward, new_state = env.apply_action(action)
                replay = Replay(prior_state, action, reward, new_state)
                self.save_replay(replay)
                mini_batch = self.sample_minibatch()
                mini_batch_targets = self.get_mini_batch_targets(mini_batch)
                mini_batch_states = [replay.post_state for replay in mini_batch]
                self.model.train_on_batch(mini_batch_states, mini_batch_targets)

                gameIsNotFinished = env.is_terminal_state()

    def save_replay(self, replay):
        self.replays.pop(0)
        self.replays.append(replay)

    def sample_minibatch(self):
        return np.random.choice(self.replays, self.mini_batch_size, replace=False)

    def get_mini_batch_targets(self, mini_batch):
        return [replay._reward if replay._reward == 1
            else self.discount * np.max(self.model.predict(replay._post_state)) for replay in mini_batch]

    @staticmethod
    def epsilon_greedy_predict_action(actions):
        if np.random.random() < epsilon:
            return np.random.randint(0, len(actions))
        else:
            return np.argmax(actions)


class Replay:
    def __init__(self, prior_state, action, reward, post_state):
        _prior_state = prior_state
        _action = action
        _reward = reward
        _post_state = post_state








