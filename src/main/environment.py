from abc import ABC, abstractmethod


class _Environment(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError

    @abstractmethod
    def is_terminal_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def is_blocked(self):
        raise NotImplementedError
