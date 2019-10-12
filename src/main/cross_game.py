from .errors import ZeroAgentIdError, ColumnIsFullError, OutOfGridError, AlreadyPlayedError


class CrossGame:

    def __init__(self):
        self._NB_COLUMNS = 7
        self._NB_ROWS = 6
        self._init_grid()
        self._init_game_history()

    def _init_grid(self):
        self._grid = [[0 for _ in range(self.get_nb_rows())] for _ in range(self.get_nb_columns())]

    def _init_game_history(self):
        self._last_player_agent_id = 0

    def get_nb_rows(self):
        return self._NB_ROWS

    def get_nb_columns(self):
        return self._NB_COLUMNS

    def put_token(self, col_index, agent_id):
        if agent_id == 0:
            raise ZeroAgentIdError()
        if agent_id == self._last_player_agent_id:
            raise AlreadyPlayedError(agent_id)
        if col_index >= self.get_nb_columns() or col_index < 0:
            raise OutOfGridError(agent_id, col_index, self.get_nb_columns())
        if self._grid[col_index][self.get_nb_rows() - 1] != 0:
            raise ColumnIsFullError(col_index)

        for i, slot in enumerate(self._grid[col_index]):
            if slot == 0:
                self._grid[col_index][i] = agent_id
                self._last_player_agent_id = agent_id
                break

    def check_vertical_victory(self, col_index, agent_id):
        return self._check_if_four_aligned_tokens(self._grid[col_index][::-1], agent_id)

    def _check_if_four_aligned_tokens(self, token_list, agent_id):
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
                left_border = max(0, col_index - 3)
                right_border = min(7, col_index + 3)
                row_index = 6 - reversed_row_id - 1
                bottom_border = max(0, row_index - 3)
                top_border = min(5, row_index + 3)
                ascending_diagonal = [self._grid[col_id][row_id] for col_id, row_id
                            in zip(range(left_border, right_border), range(bottom_border, top_border))]
                descending_diagonal  = [self._grid[col_id][row_id] for col_id, row_id
                            in zip(range(left_border, right_border), range(top_border, bottom_border, -1))]
                return (self._check_if_four_aligned_tokens(ascending_diagonal, agent_id) or
                        self._check_if_four_aligned_tokens(descending_diagonal, agent_id))

    def convert_grid_to_string(self):
        rows_list = ["|", "|", "|", "|", "|", "|"]
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def check_vertical_victory(self, col_index, agent_id):
        return self._check_if_four_aligned_tokens(self._grid[col_index][::-1], agent_id)

    def _check_if_four_aligned_tokens(self, token_list, agent_id):
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
                left_border = max(0, col_index - 3)
                right_border = min(7, col_index + 3)
                row_index = 6 - reversed_row_id - 1
                bottom_border = max(0, row_index - 3)
                top_border = min(5, row_index + 3)
                ascending_diagonal = [self._grid[col_id][row_id] for col_id, row_id
                            in zip(range(left_border, right_border), range(bottom_border, top_border))]
                descending_diagonal  = [self._grid[col_id][row_id] for col_id, row_id
                            in zip(range(left_border, right_border), range(top_border, bottom_border, -1))]
                return (self._check_if_four_aligned_tokens(ascending_diagonal, agent_id) or
                        self._check_if_four_aligned_tokens(descending_diagonal, agent_id))

    def display(self):
        print(self._convert_grid_to_string())

