from __future__ import absolute_import
import sys
sys.path.append('../main')
from dqn_agent import DQNAgent
from cross_game import CrossGame
import click


@click.command()
@click.option('--epsilon', default=0.25, help='Parameter of epsilon greedy exploration')
@click.option('--discount', default=0.95, help='Discount factor of future rewards')
@click.option('--num-episodes', default=100, help='Number of episodes for training')
@click.option('--batch-size', default=32, help='Number of samples in a training batch')
@click.option('--num-replay', default=100, help='Number of generated transitions for training')
def train_agent(epsilon, discount, num_episodes, batch_size, num_replay):
    environment = CrossGame()
    agent = DQNAgent(environment._nb_columns * environment._nb_rows,  # state space size
                     environment._nb_columns,  # action space size
                     environment,  # environment object
                     epsilon=epsilon,
                     discount=discount,
                     num_episodes=num_episodes,
                     batch_size=batch_size,
                     num_replay=num_replay)
    agent.init_replays(environment)
    agent.train(environment)

if __name__ == "__main__":
    train_agent()
