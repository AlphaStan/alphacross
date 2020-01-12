from src.main.cross_game import CrossGame
import click


@click.command()
@click.option('--choose-model', default=False, help='choose model')
def play_against_ai(choose_model):
    game = CrossGame()
    game.play_game_against_ai(choose_model=choose_model)

if __name__ == "__main__":
    play_against_ai()
