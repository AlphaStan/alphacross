import tensorflow as tf
from tensorflow.python.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from abc import ABC, abstractmethod
import numpy as np

from .dqn_agent  import dqn_mask_loss


class _Net(ABC):

    def __init__(self, n_actions, input_shape, trainable):
        super(_Net, self).__init__()
        self.n_actions = n_actions
        self.trainable = trainable
        self.input_shape = input_shape
        self.model = self.init_model()

    @abstractmethod
    def init_model(self):
        raise NotImplementedError

    @abstractmethod
    def process_input(self, x):
        raise NotImplementedError


class _DenseNet(_Net):

    def __init__(self, n_actions, input_shape, trainable):
        super(_DenseNet, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        raise NotImplementedError

    def process_input(self, x):
        return x


class _ConvNet(_Net):

    def __init__(self, n_actions, input_shape, trainable):
        super(_ConvNet, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        raise NotImplementedError

    def process_input(self, x):
        processed_input = np.zeros((x.shape[0], x.shape[1], 2))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j]:
                    processed_input[i, j, x.astype(np.int32)[i, j] - 1] = 1
        return processed_input


class CFDense(_DenseNet):

    def __init__(self, n_actions, input_shape, trainable=True):
        super(CFDense, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(24, activation='relu', trainable=self.trainable))
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFDense2(_DenseNet):

    def __init__(self, n_actions, input_shape, trainable=True):
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

    def __init__(self, n_actions, input_shape, trainable=True):
        super(CFConv1, self).__init__(n_actions, input_shape, trainable)

    def init_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(3, kernel_size=3, trainable=self.trainable, input_shape=self.input_shape))
        model.add(Dropout(0.5, trainable=self.trainable))
        model.add(Flatten())
        model.add(Dense(self.n_actions, activation='softmax', trainable=self.trainable))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model


class CFConv2(_ConvNet):

    def __init__(self, n_actions, input_shape, trainable=True):
        super(CFConv2, self).__init__(n_actions, input_shape, trainable)

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
