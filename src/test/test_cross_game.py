import pytest
from ..main import cross_game


"""
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
    game.play_game_against_human()#:test__display_grid_should_return_a_grid_with_two_tokens_when_two_tokens_were_played()
    captured = capsys.readouterr()
    actual_stdout = captured.out
    # Then
    assert actual_stdout == expected_stdout
    """


def test_new_instance_should_have_an_empty_grid_attribute():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game._nb_rows)] for _ in range(actual_game._nb_columns)]
    # When
    actual_grid = actual_game._grid
    # Then
    assert actual_grid == expected_grid


def test_apply_action_should_write_a_token_in_the_available_index():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game._nb_rows)] for _ in range(actual_game._nb_columns)]
    expected_grid[0][0] = 1
    # When
    actual_game.apply_action(0)
    actual_grid = actual_game._grid
    # Then
    assert actual_grid == expected_grid


def test_apply_action_should_write_another_token_above_a_previous_one():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = [[0 for _ in range(actual_game._nb_rows)] for _ in range(actual_game._nb_columns)]
    expected_grid[0][0] = 1
    expected_grid[0][1] = 2
    # When
    actual_game.apply_action(0)
    actual_game.apply_action(0)
    actual_grid = actual_game._grid
    # Then
    assert actual_grid == expected_grid


def test_apply_action_should_throw_exception_when_column_is_full():
    # Given
    actual_game = cross_game.CrossGame()
    # When
    for id in range(actual_game._nb_rows):
        actual_game.apply_action(0)
    # Then
    with pytest.raises(cross_game.ColumnIsFullError):
        actual_game.apply_action(0)


def test_check_vertical_victory_should_return_True_when_there_are_4_vertically_aligned_tokens_with_same_agent_id_in_column():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    for i in range(3):
        game.apply_action(col_index)
        game.apply_action(col_index + 1 + i)
    # When
    game.apply_action(col_index)
    is_vertical_victory = game._check_vertical_victory(game.get_state(), col_index, 1)
    # Then
    assert is_vertical_victory


def test_check_vertical_victory_should_return_False_when_there_are_less_than_4_vertically_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    col_index = 0
    for i in range(3):
        game.apply_action(col_index)
        game.apply_action(col_index + 1 + i)
    # When
    is_vertical_victory = game._check_vertical_victory(game.get_state(), col_index, 1)
    # Then
    assert not is_vertical_victory


def test_check_horizontal_victory_should_return_True_when_there_are_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    for i in range(3):
        game.apply_action(i)
        game.apply_action(4)
    game.apply_action(3)
    # When
    is_horizontal_victory = game._check_horizontal_victory(game.get_state(), i, 1)
    # Then
    assert is_horizontal_victory


def test_check_horizontal_victory_should_return_False_when_there_are_less_than_4_horizontally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    for i in range(3):
        game.apply_action(i)
        game.apply_action(4)
    # When
    is_horizontal_victory = game._check_horizontal_victory(game.get_state(), i, 1)
    # Then
    assert not is_horizontal_victory


def test_check_horizontal_victory_should_return_True_when_four_tokens_with_same_agent_id_are_aligned_on_left_border():
    # Given
    game = cross_game.CrossGame()
    for i in range(3, 0, -1):
        game.apply_action(i)
        game.apply_action(4)
    game.apply_action(0)
    # When
    is_horizontal_victory = game._check_horizontal_victory(game.get_state(), 0, 1)
    # Then
    assert is_horizontal_victory


def test_check_horizontal_victory_should_return_True_when_four_tokens_with_same_agent_id_are_aligned_on_right_border():
    # Given
    game = cross_game.CrossGame()
    for i in range(3, 6):
        game.apply_action(i)
        game.apply_action(0)
    game.apply_action(6)
    # When
    is_horizontal_victory = game._check_horizontal_victory(game.get_state(), 6, 1)
    # Then
    assert is_horizontal_victory


def test_check_horizontal_victory_should_return_True_when_four_tokens_with_same_agent_id_are_aligned_on_right_border():
    # Given
    game = cross_game.CrossGame()
    for i in range(3, 6):
        game.apply_action(i)
        game.apply_action(0)
    game.apply_action(6)
    # When
    is_horizontal_victory = game._check_horizontal_victory(game.get_state(), 6, 1)
    # Then
    assert is_horizontal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_with_same_agent_id_from_right():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(3)
    game.apply_action(2)
    game.apply_action(3)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 3, 1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_with_same_agent_id_from_left():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(3)
    game.apply_action(2)
    game.apply_action(3)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 0, 1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_ascending_diagonally_aligned_tokens_at_the_right_border_of_the_grid():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(3)
    game.apply_action(4)
    game.apply_action(4)
    game.apply_action(5)
    game.apply_action(5)
    game.apply_action(6)
    game.apply_action(5)
    game.apply_action(6)
    game.apply_action(6)
    game.apply_action(5)
    game.apply_action(6)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 6, 1)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_ascending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(2)
    game.apply_action(3)
    game.apply_action(3)
    game.apply_action(2)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 3, 1)
    # Then
    assert not is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_with_same_agent_id_from_right():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(1)
    game.apply_action(4)
    game.apply_action(2)
    game.apply_action(4)
    game.apply_action(3)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 3, 2)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_with_same_agent_id_from_left():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(1)
    game.apply_action(4)
    game.apply_action(2)
    game.apply_action(4)
    game.apply_action(3)
    game.apply_action(0)
    game.apply_action(0)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 0, 2)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_True_when_there_are_4_descending_diagonally_aligned_tokens_at_the_right_border_of_the_grid():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(3)
    game.apply_action(3)
    game.apply_action(4)
    game.apply_action(4)
    game.apply_action(5)
    game.apply_action(4)
    game.apply_action(0)
    game.apply_action(5)
    game.apply_action(0)
    game.apply_action(6)
    game.apply_action(3)
    game.apply_action(3)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 3, 2)
    # Then
    assert is_diagonal_victory


def test_check_diagonal_victory_should_return_False_when_there_are_less_than_4_descending_diagonally_aligned_tokens_with_same_agent_id():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(0)
    game.apply_action(1)
    game.apply_action(1)
    game.apply_action(2)
    game.apply_action(1)
    game.apply_action(4)
    game.apply_action(2)
    game.apply_action(4)
    # When
    is_diagonal_victory = game._check_diagonal_victory(game.get_state(), 2, 1)
    # Then
    assert not is_diagonal_victory


def test__display_grid_should_return_an_empty_grid_as_string_when_applied_on_new_instance():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = \
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |"
    # When
    actual_grid = actual_game._convert_grid_to_string()
    # Then
    assert expected_grid == actual_grid


def test__display_grid_should_return_a_grid_with_two_tokens_when_two_tokens_were_played():
    # Given
    actual_game = cross_game.CrossGame()
    expected_grid = \
        "| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n| | | | | | | |\n|2| | | | | | |\n|1| | | | | | |"
    # When
    actual_game.apply_action(0)
    actual_game.apply_action(0)
    actual_grid = actual_game._convert_grid_to_string()
    # Then
    assert expected_grid == actual_grid


def test_apply_action_should_throw_exception_when_player_tries_to_play_outside_the_grid():
    # Given
    actual_game = cross_game.CrossGame()
    # Then
    with pytest.raises(cross_game.OutOfGridError):
        # When
        actual_game.apply_action(10)


def test_apply_action_should_throw_exception_when_player_tries_to_play_outside_the_grid_with_negative_col_index():
    # Given
    actual_game = cross_game.CrossGame()
    # Then
    with pytest.raises(cross_game.OutOfGridError):
        # When
        actual_game.apply_action(-1)


def test_is_terminal_state_should_return_True_when_state_is_terminal():
    # Given
    game = cross_game.CrossGame()
    state = [[1, 1, 1, 1, 0, 0],
             [2, 2, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]
    # When
    is_terminal_state = game.is_terminal_state(state)
    # Then
    assert is_terminal_state


def test_is_terminal_state_should_return_False_when_state_is_not_terminal():
    # Given
    game = cross_game.CrossGame()
    state = [[1, 1, 1, 0, 0, 0],
             [2, 2, 2, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]]
    # When
    is_terminal_state = game.is_terminal_state(state)
    # Then
    assert not is_terminal_state


def test_is_blocked_should_return_False_when_the_grid_is_not_full():
    # Given
    game = cross_game.CrossGame()
    game.apply_action(1)
    game.apply_action(2)
    # When
    is_blocked = game.is_blocked()
    # Then
    assert not is_blocked


def test_is_blocked_should_return_True_when_the_grid_is_full():
    # Given
    game = cross_game.CrossGame()
    for i in range(game._nb_columns):
        for _ in range(game._nb_rows):
            game.apply_action(i)
    # When
    is_blocked = game.is_blocked()
    # Then
    assert is_blocked
