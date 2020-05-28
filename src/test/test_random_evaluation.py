import pytest
import numpy as np
import os

from ..main.evaluation.random_evaluation import RandomAgentEvaluator
from ..main.environment.cross_game import CrossGame
from ..main.models.dqn_agent import DQNAgent
from src.main.models.nets import load_net


@pytest.fixture(scope="module", autouse=True)
def evaluation_result():
    environment = CrossGame()
    agent = DQNAgent(environment)
    agent.net = load_net(os.path.join('resources', 'shared'))
    num_episodes = 3
    epsilon = 0.05
    evaluator = RandomAgentEvaluator(num_episodes, epsilon, agent, environment)
    evaluator.evaluate(0)
    return evaluator


def test_RandomAgentEvaluator_percentages_attribute_should_have_values_that_sum_to_one_when_evaluate_method_is_called(evaluation_result):
    # Given
    evaluator = evaluation_result
    expected_sum = 1
    # When
    win_percentages = evaluator._percentages
    actual_sum = win_percentages['agent_winning_percentage']\
        + win_percentages['random_agent_winning_percentage']\
        + win_percentages['draw_percentage']
    # Then
    assert abs(expected_sum - actual_sum) <= 1e-8


def test_RandomAgentEvaluator_percentages_attribute_should_have_only_positive_values_when_evaluate_method_is_called(evaluation_result):
    # Given
    evaluator = evaluation_result
    # When
    win_percentages = evaluator._percentages
    # Then
    assert np.all(np.array(list(win_percentages.values())) >= 0)
