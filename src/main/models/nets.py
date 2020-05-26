import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from abc import ABC, abstractmethod
import numpy as np
import warnings
import json
import os
import sys


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
        self.input_shape = self.get_input_shape_from_encoding(input_shape, self.encoding, self.n_players)
        self.model = self.init_model()

    @staticmethod
    def get_input_shape_from_encoding(input_shape, encoding, n_players):
        if encoding == '2d':
            if len(input_shape) == 2:
                return input_shape
            else:
                raise ValueError("Encoding is '2d' but len(input_shape) != 2")
        elif encoding == '3d':
            if len(input_shape) == 2:
                warnings.warn("Encoding is '3d', but len(input_shape) == 2")
                new_input_shape = input_shape[0], input_shape[1], n_players
                warnings.warn("Adding third dimension from n_players, new input_shape={}".format(new_input_shape))
                return new_input_shape
            elif len(input_shape) == 3:
                return input_shape
            else:
                raise ValueError("Encoding is '3d' but len(input_shape) != 2 | 3")

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    @staticmethod
    def process_input(x, encoding, n_players):
        if encoding == '3d' and len(x.shape) != 4:
            processed_input = np.zeros((x.shape[0], x.shape[1], x.shape[2], n_players))
            for player_id in [1, 2]:
                processed_input[:, :, :, player_id-1][np.nonzero(x == player_id)] = 1
            return processed_input
        else:
            return x

    def save(self, save_dir):
        self.model.save(os.path.join(save_dir, 'model.h5'))
        with open(os.path.join(save_dir, "attributes.json"), 'w') as data:
            attributes = {key: self.__dict__[key] for key in self.__dict__ if key != 'model'}
            attributes['net_name'] = self.__class__.__name__
            json.dump(attributes, data)


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
        if encoding == '2d':
            raise ValueError("Cannot instantiate CFConv1 net with encoding '2d'")
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

    def __init__(self, n_actions, input_shape, trainable,  encoding, n_players):
        if encoding == '2d':
            raise ValueError("Cannot instantiate CFConv2 net with encoding '2d'")
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


def load_model(load_dir):
    with open(os.path.join(load_dir, 'attributes.json')) as data:
        net_attributes = json.load(data)
    net_class = getattr(sys.modules[__name__], net_attributes['net_name'])
    net_instance = net_class(net_attributes['n_actions'],
                             net_attributes['input_shape'],
                             net_attributes['trainable'],
                             net_attributes['encoding'],
                             net_attributes['n_players'])
    net_instance.model = tf.keras.models.load_model(os.path.join(load_dir, 'model.h5'),
                                                    custom_objects={'dqn_mask_loss': dqn_mask_loss})
    return net_instance
