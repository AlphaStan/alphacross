import itertools
import os

import numpy as np
from tensorflow.keras.models import load_model

from ._environment import _Environment
from ..environment.errors import ColumnIsFullError, OutOfGridError
from ..models.dqn_agent import DQNAgent, dqn_mask_loss


class CrossGame(_Environment):

    def __init__(self,
                 nb_columns=7,
                 nb_rows=6,
                 final_state_reward=10,
                 non_final_state_reward=-1,
                 nb_aligned_tokens_win=4):
        super().__init__()
        self.nb_columns = nb_columns
        self.nb_rows = nb_rows
        self.final_state_reward = final_state_reward
        self.non_final_state_reward = non_final_state_reward
        self.nb_aligned_tokens_win = nb_aligned_tokens_win
        self.reset()

    def reset(self):
        self._init_grid()
        self._init_token_id()

    def _init_grid(self):
        self._grid = [[0 for _ in range(self._nb_rows)] for _ in range(self._nb_columns)]

    def _init_token_id(self):
        self.token_ids = itertools.cycle([1, 2]).__next__
        self.current_token_id = self.token_ids()

    def _toggle_token_id(self):
        self.current_token_id = self.token_ids()

    @property
    def nb_rows(self):
        return self._nb_rows

    @property
    def nb_columns(self):
        return self._nb_columns

    @property
    def nb_aligned_tokens_win(self):
        return self._nb_aligned_tokens_win

    @nb_rows.setter
    def nb_rows(self, nb_rows):
        if not isinstance(nb_rows, int) and not isinstance(nb_rows, float):
            raise ValueError("The number of rows has to be a numeric value")
        elif nb_rows < 0:
            raise ValueError("Trying to set a negative number of rows")
        else:
            self._nb_rows = nb_rows

    @nb_columns.setter
    def nb_columns(self, nb_columns):
        if not isinstance(nb_columns, int) and not isinstance(nb_columns, float):
            raise ValueError("The number of rows has to be a numeric value")
        elif nb_columns < 0:
            raise ValueError("Trying to set a negative number of rows")
        else:
            self._nb_columns = nb_columns

    @nb_aligned_tokens_win.setter
    def nb_aligned_tokens_win(self, nb_aligned_tokens_win):
        if not isinstance(nb_aligned_tokens_win, int) and not isinstance(nb_aligned_tokens_win, float):
            raise ValueError("The number of aligned tokens needed to win has to be a numeric value")
        elif nb_aligned_tokens_win < 0:
            raise ValueError("Trying to set a negative number of number of aligned tokens needed to win")
        elif nb_aligned_tokens_win > max(self.nb_rows, self._nb_columns):
            raise ValueError("The number of aligned tokens needed to win is too large for this board")
        else:
            self._nb_aligned_tokens_win = nb_aligned_tokens_win

    def play_game_against_human(self):
        number_of_rounds = 0
        while True:
            agent_has_played = False
            agent_id = self.current_token_id
            while not agent_has_played:
                try:
                    column_id = int(input("Player {}, please give the column number where you play\n".format(agent_id)))
                    self.apply_action(column_id)
                    agent_has_played = True
                except OutOfGridError:
                    print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                except ColumnIsFullError:
                    print("Player {}, column {} is full, please select another number between 0 and 6."
                          .format(agent_id, column_id))

            print(self)

            if self._is_winning_move(self.get_state(), column_id, agent_id):
                print("Congratulation player {}, you have won !".format(agent_id))
                break
            if self.is_blocked():
                print("It's a draw !")
                break

            number_of_rounds += 1

    def play_game_against_agent(self, agent):
        number_of_rounds = 0
        agent = DQNAgent(self)
        agent.model = self.choose_model(choose_model)

        if agent.model is None:
            return

        while True:
            agent_has_played = False
            agent_id = self.current_token_id
            while not agent_has_played:
                if number_of_rounds % 2 == 0:
                    try:
                        column_id = int(input("Player {}, please give the column number where you play\n".format(agent_id)))
                        self.apply_action(column_id)
                        agent_has_played = True
                    except OutOfGridError:
                        print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                    except ColumnIsFullError:
                        print("Player {}, column {} is full, please select another number between 0 and 6."
                              .format(agent_id, column_id))
                else:
                    try:
                        agent.play_action(self)
                        agent_has_played = True
                    except OutOfGridError:
                        print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                    except ColumnIsFullError:
                        print("Player {}, column {} is full, please select another number between 0 and 6."
                              .format(agent_id, column_id))

            print(self)

            if self._is_winning_move(self.get_state(), column_id, agent_id):
                print("Congratulation player {}, you have won !".format(agent_id))
                break
            if self.is_blocked():
                print("It's a draw !")
                break

            number_of_rounds += 1

    def apply_action(self, col_index):
        if col_index >= self._nb_columns or col_index < 0:
            raise OutOfGridError(self.current_token_id, col_index, self._nb_columns)
        if not self.is_legal_action(self._grid, col_index):
            raise ColumnIsFullError(col_index)

        for i, slot in enumerate(self._grid[col_index]):
            if slot == 0:
                self._grid[col_index][i] = self.current_token_id
                break

        agent_id = self.current_token_id
        self._toggle_token_id()
        if self._is_winning_move(self.get_state(), col_index, agent_id):
            return self.final_state_reward, self.get_state()
        else:
            return self.non_final_state_reward, self.get_state()

    def is_legal_action(self, state, col_index):
        return state[col_index][self._nb_rows - 1] == 0

    def is_terminal_state(self, state):
        for col_index in range(self._nb_columns):
            for agent_id in [1, 2]:
                if self._is_winning_move(state, col_index, agent_id):
                    return True
        return False

    def _is_winning_move(self, state, col_index, agent_id):
        return (self._check_vertical_victory(state, col_index, agent_id) or
                self._check_horizontal_victory(state, col_index, agent_id) or
                self._check_diagonal_victory(state, col_index, agent_id))

    def _check_vertical_victory(self, state, col_index, agent_id):
        return self._check_if_enough_aligned_tokens(state[col_index][::-1], agent_id)

    def _check_if_enough_aligned_tokens(self, token_list, agent_id):
        previous_token = 0
        consecutive_tokens = 0
        for token in token_list:
            if token == agent_id and previous_token == agent_id:
                consecutive_tokens += 1
                previous_token = token
            elif token == agent_id:
                consecutive_tokens = 1
                previous_token = token
            else:
                consecutive_tokens = 0
            if consecutive_tokens == self.nb_aligned_tokens_win:
                return True
        return False

    def _check_horizontal_victory(self, state, col_index, agent_id):
        for reversed_row_id, token in enumerate(state[col_index][::-1]):
            if token == agent_id:
                left_border = max(0, col_index - 3)
                right_border = min(self.get_n_rows(state), col_index + 3)
                row_id = self.get_n_rows(state) - reversed_row_id - 1
                row = [state[col_id][row_id] for col_id in range(left_border, right_border+1)]
                return self._check_if_enough_aligned_tokens(row, agent_id)

    def _check_diagonal_victory(self, state, col_index, agent_id):
        for reversed_row_id, token in enumerate(state[col_index][::-1]):
            if token == agent_id:
                left_border = col_index - 3
                right_border = col_index + 3
                row_index = len(state[0]) - reversed_row_id - 1
                bottom_border = row_index - 3
                top_border = row_index + 3
                ascending_diagonal = self._get_ascending_diagonal(state,
                                                                 bottom_border,
                                                                 left_border,
                                                                 right_border,
                                                                 top_border)
                descending_diagonal = self._get_descending_diagonal(state,
                                                                   bottom_border,
                                                                   left_border,
                                                                   right_border,
                                                                   top_border)
                return (self._check_if_enough_aligned_tokens(ascending_diagonal, agent_id) or
                        self._check_if_enough_aligned_tokens(descending_diagonal, agent_id))

    @classmethod
    def _get_descending_diagonal(cls, state, bottom_border, left_border, right_border, top_border):
        return [state[col_id][row_id]
                if 0 <= col_id < cls.get_n_columns(state) and 0 <= row_id < cls.get_n_rows(state)
                else 0
                for col_id, row_id
                in
                zip(range(left_border, right_border+1), range(top_border, bottom_border-1, -1))
                ]

    @classmethod
    def _get_ascending_diagonal(cls, state, bottom_border, left_border, right_border, top_border):
        return [state[col_id][row_id]
                if 0 <= col_id < cls.get_n_columns(state) and 0 <= row_id < cls.get_n_rows(state)
                else 0
                for col_id, row_id
                in zip(range(left_border, right_border+1), range(bottom_border, top_border+1))
                ]

    @staticmethod
    def get_n_rows(state):
        return len(state[0])

    @staticmethod
    def get_n_columns(state):
        return len(state)

    def _convert_grid_to_string(self):
        rows_list = ["|"] * self._nb_rows
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def __str__(self):
        return self._convert_grid_to_string() + "\n"

    def get_state(self):
        return np.array(self._grid, np.float32)

    def get_shape(self):
        return self._nb_columns, self._nb_rows

    def get_action_space_size(self):
        return self._nb_columns

    def get_state_space_size(self):
        return self._nb_columns * self._nb_rows

    def is_blocked(self):
        return 0 not in self.get_state()

    def get_current_player_id(self):
        return self.current_token_id
