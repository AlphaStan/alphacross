import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from abc import ABC, abstractmethod
import numpy as np
import warnings


def dqn_mask_loss(batch_data, y_pred):
    # The target is defined only for the action that was taken during the replay, hence the loss is computed based
    # only on this action's output
    batch_actions = tf.dtypes.cast(batch_data[:, 1], tf.int32)
    batch_true_q_values = batch_data[:, 0]
    mask = tf.one_hot(batch_actions, depth=y_pred.shape[1], dtype=tf.bool, on_value=True, off_value=False)
    batch_predicted_q_values = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.Huber()(batch_true_q_values, batch_predicted_q_values)


class _Net(ABC):

    def __init__(self, n_actions, input_shape, trainable, encoding, n_players):
        super(_Net, self).__init__()
        self.n_actions = n_actions
        self.trainable = trainable
        self.encoding = encoding
        self.n_players = n_players
        self.input_shape = self._get_input_shape_from_encoding(input_shape)
        self.model = self.init_model()

    def _get_input_shape_from_encoding(self, input_shape):
        if self.encoding == '2d':
            if len(input_shape) == 2:
                return input_shape
            else:
                raise ValueError("Encoding is '2d' but len(input_shape) != 2")
        if self.encoding == '3d':
            if len(input_shape) == 2:
                warnings.warn("Encoding is '3d', but len(input_shape) == 2")
                warnings.warn("Adding third dimension from n_players, new input_shape={}".format(self.input_shape))
                return input_shape[0], input_shape[1], self.n_players

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    def process_input(self, x):
        if self.encoding == '3d' and len(x.shape) != 4:
            processed_input = np.zeros((x.shape[0], x.shape[1], x.shape[2], self.n_players))
            for b in range(x.shape[0]):
                for i in range(x.shape[1]):
                    for j in range(x.shape[2]):
                        if x[b, i, j]:
                            processed_input[b, i, j, x.astype(np.int32)[b, i, j] - 1] = 1
            return processed_input
        else:
            return x


class CFDense(_Net):

    def __init__(self, n_actions, input_shape, trainable, encoding, n_players):
        super(CFDense, self).__init__(n_actions, input_shape, trainable, encoding, n_players)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(24, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFDense2(_Net):

    def __init__(self, n_actions, input_shape, trainable, encoding, n_players):
        super(CFDense2, self).__init__(n_actions, input_shape, trainable, encoding, n_players)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(84, activation='relu', trainable=self.trainable))
        model.add(Dense(168, activation='relu', trainable=self.trainable))
        model.add(Dense(64, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='RMSprop', metrics=['accuracy'])
        return model


class CFConv1(_Net):

    def __init__(self, n_actions, input_shape, trainable, encoding, n_players):
        super(CFConv1, self).__init__(n_actions, input_shape, trainable, encoding, n_players)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, kernel_size=3, trainable=self.trainable, input_shape=self.input_shape))
        model.add(Dropout(0.5, trainable=self.trainable))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFConv2(_Net):

    def __init__(self, n_actions, input_shape, trainable, encoding, n_players):
        super(CFConv2, self).__init__(n_actions, input_shape, trainable, encoding, n_players)

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
