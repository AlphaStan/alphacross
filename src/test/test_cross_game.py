import pytest
from ..main import cross_game


def test_new_instance_should_have_an_empty_grid_attribute():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.get_nb_rows())] for _ in range(actual_game.get_nb_columns())]
    # When
    actual_grid = actual_game._grid
    # Then
    assert (actual_grid == expected_grid)


def test_put_token_should_write_a_token_in_the_available_index():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.get_nb_rows())] for _ in range(actual_game.get_nb_columns())]
    expected_grid[0][0] = 1
    # When
    actual_game.put_token(0, 1)
    actual_grid = actual_game._grid
    # Then
    assert (actual_grid == expected_grid)


def test_put_token_should_write_another_token_above_a_previous_one():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.get_nb_rows())] for _ in range(actual_game.get_nb_columns())]
    expected_grid[0][0] = 1
    expected_grid[0][1] = 2
    # When
    actual_game.put_token(0, 1)
    actual_game.put_token(0, 2)
    actual_grid = actual_game._grid
    # Then
    assert (actual_grid == expected_grid)


def test_put_token_should_throw_exception_when_agent_id_equals_zero():
    # Given
    actual_game = cross_game.CrossGame()
    # When
    with pytest.raises(cross_game.ZeroAgentIdError):
        actual_game.put_token(0, 0)


def test_put_token_should_throw_exception_when_column_is_full():
    # Given
    actual_game = cross_game.CrossGame()
    # When
    for id in range(actual_game.get_nb_rows()):
        actual_game.put_token(0, id + 1)  # to avoid AlreadyPlayed error the agent id has to change
        # between consecutive moves
    # Then
    with pytest.raises(cross_game.ColumnIsFullError):
        actual_game.put_token(0, 1)


def test__display_grid_should_return_an_empty_grid_as_string_when_applied_on_new_instance():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid =\
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |"
    # When
    actual_grid = actual_game.convert_grid_to_string()
    # Then
    assert(expected_grid == actual_grid)


def test__display_grid_should_return_a_grid_with_two_tokens_when_two_tokens_were_played():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = \
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|2| | | | | | |\n|1| | | | | | |"
    # When
    actual_game.put_token(0, 1)
    actual_game.put_token(0, 2)
    actual_grid = actual_game.convert_grid_to_string()
    # Then
    assert (expected_grid == actual_grid)


def test_put_token_should_throw_exception_when_player_tries_to_play_outside_the_grid():
    # Given
    actual_game = cross_game.CrossGame()
    # Then
    with pytest.raises(cross_game.OutOfGridError):
        # When
        actual_game.put_token(10, 1)


def test_put_token_should_throw_exception_when_player_tries_to_play_outside_the_grid_with_negative_col_index():
    # Given
    actual_game = cross_game.CrossGame()
    # Then
    with pytest.raises(cross_game.OutOfGridError):
        # When
        actual_game.put_token(-1, 1)


def test_put_token_should_throw_exception_when_player_plays_two_times_in_a_row():
    # Given
    actual_game = cross_game.CrossGame()
    # When
    actual_game.put_token(0, 1)
    # Then
    with pytest.raises(cross_game.AlreadyPlayedError):
        actual_game.put_token(0, 1)
