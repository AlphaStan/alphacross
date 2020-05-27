import os
import traceback
import tempfile

from src.main.environment.cross_game import CrossGame
from src.train_model import train_agent
from src.play_against_agent import play_against_agent
from click.testing import CliRunner

def test_can_run_play_against_agent():
    # Given
    runner = CliRunner()
    input = "6\n6\n6\n6\n"
    path_to_models = os.path.join("ressources/", "test_can_run_play_against_agent")
    options = "--path-to-models {}".format(path_to_models)
    # When
    result = runner.invoke(play_against_agent, options, input)
    # Then
    assert result.exit_code == 0

def test_can_run_play_against_human(monkeypatch, capsys):
    # Given
    inputs = [0, 1, 0, 1, 0, 1, 0]
    input_generator = (i for i in inputs)
    monkeypatch.setattr('builtins.input', lambda prompt: next(input_generator))
    game = CrossGame()
    # When
    game.play_game_against_human()
    captured = capsys.readouterr()
    # Then
    assert captured.err == ""


def test_can_run_train_model():
    # Given
    runner = CliRunner()
    epsilon = "0.05"
    discount = "0.95"
    num_episodes = "1"
    batch_size = "1"
    num_replays = "1"
    model_name = "model"

    # When
    with tempfile.TemporaryDirectory() as tmp_dir:
        options = "--epsilon {} --discount {} --num-episodes {} --batch-size {} --num-replays {} --save-dir \"{}\" " \
                  "--model-name {}".format(epsilon, discount, num_episodes, batch_size, num_replays, tmp_dir, model_name)
        result = runner.invoke(train_agent, options)

    # Then
    assert result.exit_code == 0
