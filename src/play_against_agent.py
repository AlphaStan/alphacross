import click

from main.environment.cross_game import CrossGame
from main.models.dqn_agent import DQNAgent
from main.utils import choose_model


@click.command()
@click.option('--path-to-models', type=click.Path(exists=True), default='./models', help='Path to trained models')
@click.option('--choose', type=bool, default=False, help='If True, choose the model in terminal')
def play_against_agent(path_to_models, choose):
    game = CrossGame()
    agent = DQNAgent(game)
    model = choose_model(path_to_models, choose=choose)
    agent.model = model
    game.play_game_against_agent(agent)


if __name__ == "__main__":
    play_against_agent()
