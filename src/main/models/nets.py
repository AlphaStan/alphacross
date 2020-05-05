import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout

from .loss import dqn_mask_loss


class CFDense:

    def __init__(self, n_actions, input_shape, trainable=True):
        self.n_actions = n_actions
        self.trainable = trainable
        self.input_shape = input_shape

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(24, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFConv1:

    def __init__(self, n_actions, input_shape, trainable=True):
        self.n_actions = n_actions
        self.trainable = trainable
        self.input_shape = input_shape

    def get_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, kernel_size=3, trainable=self.trainable, input_shape=self.input_shape))
        model.add(Dropout(0.5, trainable=self.trainable))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model
