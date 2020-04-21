import click

from main.environment.cross_game import CrossGame
from main.constants import PATH_TO_MODELS


@click.command()
@click.option('--model-path', type=click.Path(exists=True), default=PATH_TO_MODELS, help='choose model')
def play_against_ai(choose_model):
    game = CrossGame()
    game.play_game_against_ai(choose_model=model_path)

if __name__ == "__main__":
    play_against_ai()
