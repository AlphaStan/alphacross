import numpy as np
import tensorflow as tf
from src.main.cross_game import CrossGame

epsilon = 0.05

c = CrossGame()
nb_column = c.nb_columns

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(c.nb_rows * c.nb_columns, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(c.nb_columns, activation=tf.keras.activations.softmax))

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


def init_replays():
    replays = []
    while len(replays) < 1000:
        replays += get_replay()
    return replays

def get_replay():
    replays = []
    gameIsNotFinished = True
    counter = 0
    c = CrossGame()
    while(gameIsNotFinished):
        prior_state = c.get_flat_grid()
        action = epsilon_greedy_predict_action()
        reward  = c.play_token(action, counter%2)
        post_state = c.get_flat_grid()
        replays.append(Replay(prior_state, action, reward, post_state))
        counter += 1

def epsilon_greedy_predict_action(state,  model):
    if np.random.random() < epsilon:
        return np.random.randint(0, nb_column)
    else:
        return model.predict(state)

def train_model():
    gameIsNotFinished = True
    counter = 0
    c = CrossGame()
    while(gameIsNotFinished):
        action = model.epsilon_greedy_predict_action()
        reward = c.play_token(action, counter%2)
        counter += 1


class Replay:
    def __init__(self, prior_state, action, reward, post_state):
        _prior_state = prior_state
        _action = action
        _reward = reward
        _post_state = post_state








