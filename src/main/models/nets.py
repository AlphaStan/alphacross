import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from abc import ABC, abstractmethod
import numpy as np


def dqn_mask_loss(batch_data, y_pred):
    # The target is defined only for the action that was taken during the replay, hence the loss is computed based
    # only on this action's output
    batch_actions = tf.dtypes.cast(batch_data[:, 1], tf.int32)
    batch_true_q_values = batch_data[:, 0]
    mask = tf.one_hot(batch_actions, depth=y_pred.shape[1], dtype=tf.bool, on_value=True, off_value=False)
    batch_predicted_q_values = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.Huber()(batch_true_q_values, batch_predicted_q_values)


class _DenseNet(ABC):

    def __init__(self, n_actions, input_shape, trainable):
        super(_DenseNet, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.trainable = trainable
        self.model = self.init_model()

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    @staticmethod
    def process_input(x):
        return x


class _ConvNet(ABC):

    def __init__(self, n_actions, input_shape, trainable, n_players):
        super(_ConvNet, self).__init__()
        self.n_actions = n_actions
        self.input_shape = input_shape[0], input_shape[1], n_players
        self.trainable = trainable
        self.n_players = n_players
        self.model = self.init_model()

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    def process_input(self, x):
        processed_input = np.zeros((x.shape[0], x.shape[1], x.shape[2], self.n_players))
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                for j in range(x.shape[2]):
                    if x[b, i, j]:
                        processed_input[b, i, j, x.astype(np.int32)[b, i, j] - 1] = 1
        return processed_input


class CFDense(_DenseNet):

    def __init__(self, n_actions, input_shape, trainable, *args):
        super(CFDense, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(24, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFDense2(_DenseNet):

    def __init__(self, n_actions, input_shape, trainable, *args):
        super(CFDense2, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(84, activation='relu', trainable=self.trainable))
        model.add(Dense(168, activation='relu', trainable=self.trainable))
        model.add(Dense(64, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='RMSprop', metrics=['accuracy'])
        return model


class CFConv1(_ConvNet):

    def __init__(self, n_actions, input_shape, trainable, n_players):
        super(CFConv1, self).__init__(n_actions, input_shape, trainable, n_players)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, kernel_size=3, trainable=self.trainable, input_shape=self.input_shape))
        model.add(Dropout(0.5, trainable=self.trainable))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFConv2(_ConvNet):

    def __init__(self, n_actions, input_shape, trainable, n_players):
        super(CFConv2, self).__init__(n_actions, input_shape, trainable, n_players)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(4, kernel_size=3, padding='same', trainable=self.trainable, input_shape=self.input_shape))
        model.add(BatchNormalization(trainable=self.trainable))
        model.add(Conv2D(16, kernel_size=3, padding='same', trainable=self.trainable))
        model.add(BatchNormalization(trainable=self.trainable))
        model.add(Conv2D(32, kernel_size=3, padding='same', trainable=self.trainable))
        model.add(BatchNormalization(trainable=self.trainable))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='RMSprop', metrics=['accuracy'])
        return model
