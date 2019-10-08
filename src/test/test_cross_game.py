import pytest
from ..main import cross_game


def test_new_instance_should_have_an_empty_grid_attribute():
    ## Given
    expected_grid = [[0 for _ in range(6)] for _ in range(7)]
    ## When
    actual_game = cross_game.CrossGame()
    actual_grid = actual_game._grid
    ## Then
    assert(actual_grid == expected_grid)

def test_put_token_should_write_a_token_in_the_available_index():
    ## Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(6)] for _ in range(7)]
    expected_grid[0][0] = 1
    ## When
    actual_game.put_token(0, 1)
    actual_grid = actual_game._grid
    ## Then
    assert(actual_grid == expected_grid)

def test_put_token_should_write_another_token_above_a_previous_one():
    ## Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(6)] for _ in range(7)]
    expected_grid[0][0] = 1
    expected_grid[0][1] = 2
    ## When
    actual_game.put_token(0, 1)
    actual_game.put_token(0, 2)
    actual_grid = actual_game._grid
    ## Then
    assert(actual_grid == expected_grid)

def test_put_token_should_throw_exception_when_agent_id_equals_zero():
    ## Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(6)] for _ in range(7)]
    ## When
    with pytest.raises(cross_game.ZeroAgentIdError):
        actual_game.put_token(0, 0)

def test_put_token_should_throw_exception_when_column_is_full():
    ## Given
    actual_game = cross_game.CrossGame()
    ## When
    for _ in range(6):
        actual_game.put_token(0, 1)
    ## Then
    with pytest.raises(cross_game.ColumnIsFullError):
        actual_game.put_token(0, 1)


def test__display_grid_should_return_an_empty_grid_as_string_when_applied_on_new_instance():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid =\
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |"
    # When
    actual_grid = actual_game._display_grid()
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
    actual_grid = actual_game._display_grid()
    # Then
    assert (expected_grid == actual_grid)