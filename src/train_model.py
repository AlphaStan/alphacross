import click

from src.main.environment.cross_game import CrossGame
from src.main.models.dqn_agent import DQNAgent


@click.command()
@click.option('--epsilon', type=float, default=0.25, help='Parameter of epsilon greedy exploration')
@click.option('--net-name', default='CFConv2',
              type=click.Choice(['CFDense', 'CFDense2', 'CFConv1', 'CFConv2'], case_sensitive=False))
@click.option('--encoding', type=click.Choice(['2d', '3d'], case_sensitive=False),
              default='3d', help="Number of players to simulate")
@click.option('--n-players', type=int, default=2)
@click.option('--discount', type=float, default=0.95, help='Discount factor of future rewards')
@click.option('--num-episodes', type=int, default=100, help='Number of episodes for training')
@click.option('--batch-size', type=int, default=32, help='Number of samples in a training batch')
@click.option('--num-replays', type=int, default=100, help='Number of generated transitions for training')
@click.option('--save-dir', type=click.Path(exists=True), default='./models',
              help="Directory to save the model outputs")
@click.option('--model_name', type=str, default="")
def train_agent(epsilon, net_name, encoding, n_players, discount, num_episodes, batch_size, num_replays, save_dir, model_name):
>>>>>>> add model name handling
    environment = CrossGame()
    agent = DQNAgent(environment,  # environment object
                     epsilon=epsilon,
                     net_name=net_name,
                     encoding=encoding,
                     n_players=n_players,
                     discount=discount,
                     num_episodes=num_episodes,
                     batch_size=batch_size,
                     num_replays=num_replays,
                     save_dir=save_dir,
                     model_name=model_name)
    agent.init_replays(environment)
    agent.train(environment)


if __name__ == "__main__":
    train_agent()
