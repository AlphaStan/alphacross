from abc import ABC, abstractmethod


class _Environment(ABC):

    def __init__(self):
        super().__init__()

    def evaluate_state_action(self, state, action):
        raise NotImplementedError

    def evaluate_state(self, state):
        raise NotImplementedError

    @abstractmethod
    def apply_action(self, action):
        raise NotImplementedError

    @abstractmethod
    def get_state(self):
        raise NotImplementedError
