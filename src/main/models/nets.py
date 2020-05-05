import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout

from .loss import dqn_mask_loss


class CFDense:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_actions, activation='softmax'))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFConv1:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, kernel_size=3))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax'))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model
