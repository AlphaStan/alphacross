import click

from main.environment.cross_game import CrossGame
from main.models.dqn_agent import DQNAgent
from main.utils import choose_model


@click.command()
def play_against_ai():
    game = CrossGame()
    agent = DQNAgent(game)
    model = choose_model(True)
    agent.model = model
    game.play_game_against_ai(agent)


if __name__ == "__main__":
    play_against_ai()
