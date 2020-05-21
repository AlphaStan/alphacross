from tensorflow.keras.models import load_model
import pytest
import numpy as np

from ..main.evaluation.random_evaluation import RandomAgentEvaluator
from ..main.environment.cross_game import CrossGame
from ..main.models.dqn_agent import dqn_mask_loss, DQNAgent


@pytest.fixture(scope="module", autouse=True)
def evaluation_result():
    environment = CrossGame()
    model = load_model('./models/trained_model_15122019_234912.h5', custom_objects={'dqn_mask_loss': dqn_mask_loss})
    agent = DQNAgent(environment)
    agent.model = model
    num_episodes = 3
    epsilon = 0.05
    evaluator = RandomAgentEvaluator(num_episodes, epsilon, agent, environment)
    evaluator.evaluate()
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
