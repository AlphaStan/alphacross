from src.models.agent import _Agent
import numpy as np


class RandomAgent(_Agent):

    def __init__(self, action_space_size):
        super().__init__()
        self.action_space_size = action_space_size

    def play_action(self, env):
        action_id = np.random.choice(self.action_space_size)
        env.apply_action(action_id)

    def train(self):
        pass
