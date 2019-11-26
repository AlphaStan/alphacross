from .dqn_agent import DQNAgent
from src.main.cross_game import CrossGame


def train_agent():
    environment = CrossGame()
    agent = DQNAgent(environment.nb_columns * environment.nb_rows, environment.nb_columns)
    agent.train(environment)
