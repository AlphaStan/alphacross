import os
import pytest
from src.train_model import train_agent
from click.testing import CliRunner

def test_can_run_play_against_agent():
    # Given
    # When
    # Then
    assert True is True

def test_can_run_play_against_human():
    # Given
    # When
    # Then
    assert True is True

def test_can_run_train_model():
    # Given
    runner = CliRunner()
    epsilon = 0.05
    discount = 0.95
    num_episodes = 1
    batch_size = 1
    num_replays = 1
    save_dir = "./models"
    model_name = "test"
    # When
    if os.path.exists("./models/test"):
        os.remove("./models/test")
    result = runner.invoke(train_agent, [epsilon, discount, num_episodes, batch_size, num_replays, save_dir, model_name])
    """
    train_agent(epsilon, discount, num_episodes, batch_size, num_replays, save_dir, model_name)
    model_exists = os.path.exists("./models/test.h5")
    """
    # Then
    assert result.exit_code == 0
