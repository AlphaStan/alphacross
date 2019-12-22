from src.models.dqn_agent import DQNAgent
from src.main.cross_game import CrossGame


def train_agent():
    environment = CrossGame()
    agent = DQNAgent(environment._nb_columns * environment._nb_rows, environment._nb_columns, environment,
                     batch_size=32, num_episodes=1000, num_replay=10000)
    agent.init_replays(environment)
    agent.train(environment)

if __name__ == "__main__":
    train_agent()
