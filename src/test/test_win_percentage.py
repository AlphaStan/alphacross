from tensorflow.python.keras.models import load_model
import numpy as np

from ..main.evaluation.win_percentage import RandomAgentEvaluator
from ..main.environment.cross_game import CrossGame
from ..main.models.dqn_agent import dqn_mask_loss, DQNAgent


def test_RandomAgentEvaluator_percentages_attribute_should_have_values_that_sum_to_one_when_evaluate_method_is_called():
    # Given
    environment = CrossGame()
    model = load_model('./models/trained_model_15122019_234912.h5', custom_objects={'dqn_mask_loss': dqn_mask_loss})
    agent = DQNAgent(environment)
    agent.model = model
    num_episodes = 10
    expected_sum = 1
    evaluator = RandomAgentEvaluator(agent, environment, num_episodes)
    # When
    evaluator.evaluate()
    win_percentages = evaluator._percentages
    actual_sum = win_percentages['agent_winning_percentage']\
        + win_percentages['random_agent_winning_percentage']\
        + win_percentages['draw_percentage']
    # Then
    assert abs(expected_sum - actual_sum) <= 1e-8


def test_RandomAgentEvaluator_evaluate_method_should_return_the_percentage_of_victories_of_a_given_agent():
    # Given
    np.random.seed(42)
    environment = CrossGame()
    model = load_model('./models/trained_model_15122019_234912.h5', custom_objects={'dqn_mask_loss': dqn_mask_loss})
    agent = DQNAgent(environment)
    agent.model = model
    num_episodes = 10
    expected_win_percentage = 0.7
    evaluator = RandomAgentEvaluator(agent, environment, num_episodes)
    # When
    actual_win_percentage = evaluator.evaluate()
    # Then
    assert expected_win_percentage == actual_win_percentage
