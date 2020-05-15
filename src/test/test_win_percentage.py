import pytest
from tensorflow.python.keras.models import load_model
import numpy as np

from ..main.evaluation.win_percentage import against_random_agent
from ..main.environment.cross_game import CrossGame
from ..main.models.dqn_agent import dqn_mask_loss, DQNAgent


def test_against_randon_agent_should_return_a_dict_with_win_percentages_of_each_agents_and_draws():
    # Given
    np.random.seed(42)
    model = load_model('./models/trained_model_15122019_234912.h5', custom_objects={'dqn_mask_loss': dqn_mask_loss})
    environment = CrossGame()
    agent = DQNAgent(environment)
    agent.model = model
    num_episodes = 10
    expected_sum = 1
    # When
    win_percentages = against_random_agent(agent, environment, num_episodes)
    actual_sum = win_percentages['agent_winning_percentage']\
        + win_percentages['random_agent_winning_percentage']\
        + win_percentages['draw_percentage']
    # Then
    assert expected_sum == actual_sum


