import os
import traceback

from src.main.environment.cross_game import CrossGame
from src.train_model import train_agent
from click.testing import CliRunner

def test_can_run_play_against_agent():
    # Given
    # When
    # Then
    assert True is True

def test_can_run_play_against_human(monkeypatch, capsys):
    # Given
    inputs = [0, 1, 0, 1, 0, 1, 0]
    input_generator = (i for i in inputs)
    monkeypatch.setattr('builtins.input', lambda prompt: next(input_generator))
    game = CrossGame()
    game.play_game_against_human()
    captured = capsys.readouterr()

    # When
    # Then

def test_can_run_train_model():
    # Given
    runner = CliRunner(mix_stderr=False)
    epsilon = "0.05"
    discount = "0.95"
    num_episodes = "1"
    batch_size = "1"
    num_replays = "1"
    save_dir = "./models"
    model_name = "test"
    options = "--epsilon {} --discount {} --num-episodes {} --batch-size {} --num-replays {} --save-dir {} " \
              "--model-name {}".format(epsilon, discount, num_episodes, batch_size, num_replays, save_dir, model_name)
    # When
    path = "./models/test"
    if os.path.exists(path):
        for file in os.listdir(path):
            os.remove(path + "/" + file)
        os.rmdir("./models/test")
    result = runner.invoke(train_agent, options)
    # Then
    assert result.exit_code == 0
