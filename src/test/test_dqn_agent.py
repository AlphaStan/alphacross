from tensorflow.keras.models import load_model

from src.main.models.dqn_agent import DQNAgent
from src.main.environment.cross_game import CrossGame
from src.main.models.nets import dqn_mask_loss


def test_select_action_should_always_select_the_same_action_given_the_same_environment_and_same_seed():
    # Given
    seed = 42
    model = load_model('./models/trained_model_15122019_234912.h5', custom_objects={'dqn_mask_loss': dqn_mask_loss})
    environment = CrossGame()
    agent = DQNAgent(environment)
    agent.net.model = model
    agent.net.encoding = '2d'
    expected_selected_action = 2
    environment.apply_action(6)
    # When
    actual_selected_action = agent.select_action(environment, seed=seed)
    # Then
    assert actual_selected_action == expected_selected_action
