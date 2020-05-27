from abc import ABC, abstractmethod


class _Agent(ABC):

    def __init__(self):
        super().__init__()
        self.model = None

    @abstractmethod
    def play_action(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError
