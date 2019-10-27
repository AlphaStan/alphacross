import pytest

from ..main import cross_game


def test_play_should_display_victory1_when_player1_wins(monkeypatch, capsys):
    # Given
    inputs = [0, 1, -3, 0, 1, 0, 1, 0]
    input_generator = (i for i in inputs)
    monkeypatch.setattr('builtins.input', lambda prompt: next(input_generator))
    game = cross_game.CrossGame()
    expected_stdout = '\n'.join(
        ['| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1| | | | | | |\n',
         '| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1|2| | | | | |\n',
         'Player 1, you should give a number between 0 and 6.',
         '| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1| | | | | | |\n|1|2| | | | | |\n',
         '| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n',
         '| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1| | | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n',
         '| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n',
         '| | | | | | | |\n| | | | | | | |\n|1| | | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n|1|2| | | | | |\n',
         'Congratulation player 1, you have won !\n'
         ])
    # When
    game.play()
    captured = capsys.readouterr()
    actual_stdout = captured.out
    # Then
    assert actual_stdout == expected_stdout


def test_new_instance_should_have_an_empty_grid_attribute():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.nb_rows)] for _ in range(actual_game.nb_columns)]
    # When
    actual_grid = actual_game._grid
    # Then
    assert (actual_grid == expected_grid)


def test_put_token_should_write_a_token_in_the_available_index():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.nb_rows)] for _ in range(actual_game.nb_columns)]
    expected_grid[0][0] = 1
    # When
    actual_game.put_token(0, 1)
    actual_grid = actual_game._grid
    # Then
    assert (actual_grid == expected_grid)


def test_put_token_should_write_another_token_above_a_previous_one():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game.nb_rows)] for _ in range(actual_game.nb_columns)]
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
    for id in range(actual_game.nb_rows):
        actual_game.put_token(0, id + 1)  # to avoid AlreadyPlayed error the agent id has to change
        # between consecutive moves
    # Then
    with pytest.raises(cross_game.ColumnIsFullError):
        actual_game.put_token(0, 1)


def test_check_vertical_victory_should_return_True_when_there_are_4_vertically_aligned_tokens_with_same_agent_id_in_column():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    agent_id_1 = 1
    agent_id_2 = 2
    for i in range(3):
        game.put_token(col_index, agent_id_1)
        game.put_token(col_index + 1 + i, agent_id_2)
    # When
    game.put_token(col_index, agent_id_1)
    is_vertical_victory = game.check_vertical_victory(col_index, agent_id_1)
    # Then
    assert is_vertical_victory


def test_check_vertical_victory_should_return_False_when_there_are_less_than_4_vertically_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    agent_id_1 = 1
    agent_id_2 = 2
    for i in range(3):
        game.put_token(col_index, agent_id_1)
        game.put_token(col_index + 1 + i, agent_id_2)
    # When
    is_vertical_victory = game.check_vertical_victory(col_index, agent_id_1)
    # Then
    assert (not is_vertical_victory)


def test_check_horizontal_victory_should_return_True_when_there_are_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    for i in range(3):
        game.put_token(i, agent_id_1)
        game.put_token(4, agent_id_2)
    game.put_token(3, agent_id_1)
    # When
    is_horizontal_victory = game.check_horizontal_victory(i, agent_id_1)
    # Then
    assert is_horizontal_victory


def test_check_horizontal_victory_should_return_False_when_there_are_less_than_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    for i in range(3):
        game.put_token(i, agent_id_1)
        game.put_token(4, agent_id_2)
    # When
    is_horizontal_victory = game.check_horizontal_victory(i, agent_id_1)
    # Then
    assert (not is_horizontal_victory)


def test_check_horizontal_victory_should_return_True_when_four_tokens_with_same_agent_id_are_aligned():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    for i in range(3, 0, -1):
        game.put_token(i, agent_id_1)
        game.put_token(4, agent_id_2)
    game.put_token(0, agent_id_1)
    # When
    is_horizontal_victory = game.check_horizontal_victory(i, agent_id_1)
    # Then
    assert is_horizontal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_with_same_agent_id_from_right():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(3, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(3, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(3, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_with_same_agent_id_from_left():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(3, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(3, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(0, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_at_the_right_border_of_the_grid():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(3, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(4, agent_id_1)
    game.put_token(5, agent_id_2)
    game.put_token(5, agent_id_1)
    game.put_token(6, agent_id_2)
    game.put_token(5, agent_id_1)
    game.put_token(6, agent_id_2)
    game.put_token(6, agent_id_1)
    game.put_token(5, agent_id_2)
    game.put_token(6, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(6, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_ascending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(3, agent_id_1)
    game.put_token(2, agent_id_2)
    # When
    is_diagonal_victory = game.check_diagonal_victory(3, agent_id_1)
    # Then
    assert (not is_diagonal_victory)


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_with_same_agent_id_from_right():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(3, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(3, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_with_same_agent_id_from_left():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(3, agent_id_1)
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(0, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_at_the_right_border_of_the_grid():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(3, agent_id_2)
    game.put_token(3, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(4, agent_id_1)
    game.put_token(5, agent_id_2)
    game.put_token(4, agent_id_1)
    game.put_token(0, agent_id_2)
    game.put_token(5, agent_id_1)
    game.put_token(0, agent_id_2)
    game.put_token(6, agent_id_1)
    game.put_token(3, agent_id_2)
    game.put_token(3, agent_id_1)
    # When
    is_diagonal_victory = game.check_diagonal_victory(3, agent_id_1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_descending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    agent_id_1 = 1
    agent_id_2 = 2
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    game.put_token(0, agent_id_2)
    game.put_token(0, agent_id_1)
    game.put_token(1, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(2, agent_id_2)
    game.put_token(1, agent_id_1)
    game.put_token(4, agent_id_2)
    game.put_token(2, agent_id_1)
    game.put_token(4, agent_id_2)
    # When
    is_diagonal_victory = game.check_diagonal_victory(2, agent_id_1)
    # Then
    assert (not is_diagonal_victory)


def test__display_grid_should_return_an_empty_grid_as_string_when_applied_on_new_instance():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = \
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |"
    # When
    actual_grid = actual_game.convert_grid_to_string()
    # Then
    assert (expected_grid == actual_grid)


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
