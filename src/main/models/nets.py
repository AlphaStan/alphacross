import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense

from .loss import dqn_mask_loss


class CFDense:

    def __init__(self, env_shape, n_actions):
        self.env_shape = env_shape
        self.n_actions = n_actions

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.env_shape))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='softmax'))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model
