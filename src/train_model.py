import click

from main.environment.cross_game import CrossGame
from main.models.dqn_agent import DQNAgent


@click.command()
@click.option('--epsilon', type=float, default=0.25, help='Parameter of epsilon greedy exploration')
@click.option('--discount', type=float, default=0.95, help='Discount factor of future rewards')
@click.option('--num-episodes', type=int, default=100, help='Number of episodes for training')
@click.option('--batch-size', type=int, default=32, help='Number of samples in a training batch')
@click.option('--num-replays', type=int, default=100, help='Number of generated transitions for training')
@click.option('--save-dir', type=click.Path(exists=True), default='../../models',
              help="Directory to save the model outputs")
def train_agent(epsilon, discount, num_episodes, batch_size, num_replays, save_dir):
    environment = CrossGame()
    agent = DQNAgent(environment,  # environment object
                     epsilon=epsilon,
                     discount=discount,
                     num_episodes=num_episodes,
                     batch_size=batch_size,
                     num_replays=num_replays,
                     save_dir=save_dir)
    agent.init_replays(environment)
    agent.train(environment)


if __name__ == "__main__":
    train_agent()
