
class CrossGame:
    
    def __init__(self):
        self._grid = [[0 for _ in range(6)] for _ in range(7)]

    def put_token(self, col_index, agent_id):
        if agent_id == 0:
            raise ZeroAgentIdError()
        if self._grid[col_index][5] != 0:
            raise ColumnIsFullError(col_index)
        for i, slot in enumerate(self._grid[col_index]):
            if slot == 0:
                self._grid[col_index][i] = agent_id
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


class ZeroAgentIdError(Exception):
    def __init__(self):
        self.message = "Agent id should not be equal zero which is the convention for empty cell."

class ColumnIsFullError(Exception):
    def __init__(self, col_index):
        self.message = "You cannot play in column {} because it is full.".format(col_index)