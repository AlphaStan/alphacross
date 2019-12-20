from src.main.cross_game import CrossGame


def play_against_human():
    game = CrossGame()
    game.play_game_against_human()

if __name__ == "__main__":
    play_against_human()