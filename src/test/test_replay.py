from src.main.models.replay import Replay


def test_classmethod_toggle_state_should_toggle_player_ids_in_grid():
    # GIVEN
    state = [[1, 1, 1], [2, 2, 1], [0, 1, 2]]
    expected_state = [[2, 2, 2], [1, 1, 2], [0, 2, 1]]
    # WHEN
    actual_state = Replay.toggle_state(state)
    # THEN
    assert actual_state == expected_state
