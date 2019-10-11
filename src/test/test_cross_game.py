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


def test_check_vertical_victory_should_return_True_when_there_are_4_vertically_aligned_tokens_with_same_agent_id_in_column():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    agent_id = 1
    for _ in range(3):
        game.put_token(col_index, agent_id)
    # When
    game.put_token(col_index, agent_id)
    is_vertical_victory = game.check_vertical_victory(col_index, agent_id)
    # Then
    assert(is_vertical_victory)


def test_check_vertical_victory_should_return_False_when_there_are_less_than_4_vertically_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    agent_id = 1
    for _ in range(3):
        game.put_token(col_index, agent_id)
    # When
    is_vertical_victory = game.check_vertical_victory(col_index, agent_id)
    # Then
    assert(not is_vertical_victory)


def test_check_horizontal_victory_should_return_True_when_there_are_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i in range(4):
        game.put_token(i, agent_id)
    # When
    is_horizontal_victory = game.check_horizontal_victory(i, agent_id)
    # Then
    assert(is_horizontal_victory)


def test_check_horizontal_victory_should_return_False_when_there_are_less_than_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i in range(3):
        game.put_token(i, agent_id)
    # When
    is_horizontal_victory = game.check_horizontal_victory(i, agent_id)
    # Then
    assert(not is_horizontal_victory)


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i in range(4):
        game._grid[i][i] = agent_id
    # When
    is_diagonal_victory = game.check_diagonal_victory(i, agent_id)
    # Then
    assert(is_diagonal_victory)


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_ascending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i in range(3):
        game._grid[i][i] = agent_id
    # When
    is_diagonal_victory = game.check_diagonal_victory(i, agent_id)
    # Then
    assert(not is_diagonal_victory)


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i, j in zip(range(4), range(4, 0, -1)):
        game._grid[i][j] = agent_id
    # When
    is_diagonal_victory = game.check_diagonal_victory(i, agent_id)
    # Then
    assert(is_diagonal_victory)


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_descending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id = 1
    for i, j in zip(range(3), range(3, 0, -1)):
        game._grid[i][j] = agent_id
    # When
    is_diagonal_victory = game.check_diagonal_victory(i, agent_id)
    # Then
    assert(not is_diagonal_victory)