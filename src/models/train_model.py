from src.models.dqn_agent import DQNAgent
from src.main.cross_game import CrossGame


def train_agent():
    env = CrossGame()
    agent = DQNAgent(mini_batch_size=32)
    agent.train(env)

if __name__ == "__main__":
    train_agent()
