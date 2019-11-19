import itertools
from .errors import ColumnIsFullError, OutOfGridError
from .environment import _Environment


class CrossGame(_Environment):

    def __init__(self):
        self._NB_COLUMNS = 7
        self._NB_ROWS = 6
        self._init_grid()
        self._init_token_id()

    def _init_grid(self):
        self._grid = [[0 for _ in range(self.nb_rows)] for _ in range(self.nb_columns)]

    def _init_token_id(self):
        self.token_ids = itertools.cycle([1, 2]).__next__
        self.current_token_id = self.token_ids()

    def _toggle_token_id(self):
        self.current_token_id = self.token_ids()

    @property
    def nb_rows(self):
        return self._NB_ROWS

    @property
    def nb_columns(self):
        return self._NB_COLUMNS

    @nb_rows.setter
    def nb_rows(self, nb_rows):
        if not isinstance(nb_rows, int) or not isinstance(nb_rows, float):
            raise ValueError("The number of rows has to be a numeric value")
        elif nb_rows < 0:
            raise ValueError("Trying to set a negative number of rows")
        else:
            self._NB_ROWS = nb_rows

    @nb_columns.setter
    def nb_columns(self, nb_columns):
        if not isinstance(nb_columns, int) or not isinstance(nb_columns, float):
            raise ValueError("The number of rows has to be a numeric value")
        elif nb_columns < 0:
            raise ValueError("Trying to set a negative number of rows")
        else:
            self._NB_ROWS = nb_columns

    def play(self):
        number_of_rounds = 0
        number_of_cells = self.nb_columns * self.nb_columns
        while number_of_rounds < number_of_cells:

            agent_has_played = False
            agent_id = self.current_token_id
            while not agent_has_played:
                try:
                    column_id = input("Player {}, please give the column number where you play".format(agent_id))
                    self.apply_action(column_id)
                    agent_has_played = True
                except OutOfGridError:
                    print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                except ColumnIsFullError:
                    print("Player {}, column {} is full, please select another number between 0 and 6."
                          .format(agent_id, column_id))

            self.display()
            if self.is_winning_move(column_id, agent_id):
                print("Congratulation player {}, you have won !".format(agent_id))
                break

            number_of_rounds += 1

    def apply_action(self, col_index):
        if col_index >= self.nb_columns or col_index < 0:
            raise OutOfGridError(self.current_token_id, col_index, self.nb_columns)
        if self._grid[col_index][self.nb_rows - 1] != 0:
            raise ColumnIsFullError(col_index)

        for i, slot in enumerate(self._grid[col_index]):
            if slot == 0:
                self._grid[col_index][i] = self.current_token_id
                break
        self._toggle_token_id()

    def is_winning_move(self, col_index, agent_id):
        return (self.check_vertical_victory(col_index, agent_id) or
                self.check_horizontal_victory(col_index, agent_id) or
                self.check_diagonal_victory(col_index, agent_id))

    def check_vertical_victory(self, col_index, agent_id):
        return self._check_if_four_aligned_tokens(self._grid[col_index][::-1], agent_id)

    @staticmethod
    def _check_if_four_aligned_tokens(token_list, agent_id):
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
            if consecutive_tokens == 4:
                return True
        return False

    def check_horizontal_victory(self, col_index, agent_id):
        for reversed_row_id, token in enumerate(self._grid[col_index][::-1]):
            if token == agent_id:
                left_border = max(0, col_index - 3)
                right_border = min(6, col_index + 3)
                row_id = 6 - reversed_row_id - 1
                row = [self._grid[col_id][row_id] for col_id in range(left_border, right_border)]
                return self._check_if_four_aligned_tokens(row, agent_id)

    def check_diagonal_victory(self, col_index, agent_id):
        for reversed_row_id, token in enumerate(self._grid[col_index][::-1]):
            if token == agent_id:
                left_border = col_index - 3
                right_border = col_index + 3
                row_index = 6 - reversed_row_id - 1
                bottom_border = row_index - 3
                top_border = row_index + 3
                ascending_diagonal = self._get_ascending_diagonal(bottom_border,
                                                                  left_border,
                                                                  right_border,
                                                                  top_border)
                descending_diagonal = self._get_descending_diagonal(bottom_border,
                                                                    left_border,
                                                                    right_border,
                                                                    top_border)
                return (self._check_if_four_aligned_tokens(ascending_diagonal, agent_id) or
                        self._check_if_four_aligned_tokens(descending_diagonal, agent_id))

    def _get_descending_diagonal(self, bottom_border, left_border, right_border, top_border):
        return [self._grid[col_id][row_id]
                if 0 <= col_id < self.nb_columns and 0 <= row_id < self.nb_rows
                else 0
                for col_id, row_id
                in
                zip(range(left_border, right_border+1), range(top_border, bottom_border-1, -1))
                ]

    def _get_ascending_diagonal(self, bottom_border, left_border, right_border, top_border):
        return [self._grid[col_id][row_id]
                if 0 <= col_id < self.nb_columns and 0 <= row_id < self.nb_rows
                else 0
                for col_id, row_id
                in zip(range(left_border, right_border+1), range(bottom_border, top_border+1))
                ]

    def convert_grid_to_string(self):
        rows_list = ["|", "|", "|", "|", "|", "|"]
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def display(self):
        print(self.convert_grid_to_string())
        print()

    def get_state(self):
        return self._grid
