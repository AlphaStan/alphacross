from abc import ABC, abstractmethod


class _Agent(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def play_method(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError
