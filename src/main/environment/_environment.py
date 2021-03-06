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

    @abstractmethod
    def get_shape(self):
        raise NotImplementedError

    @abstractmethod
    def get_state_space_size(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_space_size(self):
        raise NotImplementedError

    @abstractmethod
    def play_game_against_human(self):
        raise NotImplementedError

    @abstractmethod
    def play_game_against_agent(self, agent):
        raise NotImplementedError

    @abstractmethod
    def get_current_player_id(self):
        raise NotImplementedError
