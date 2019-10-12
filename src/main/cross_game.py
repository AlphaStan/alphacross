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

    def _convert_grid_to_string(self):
        rows_list = ["|", "|", "|", "|", "|", "|"]
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def display(self):
        print(self._convert_grid_to_string())


class ZeroAgentIdError(Exception):
    def __init__(self):
        self.message = "Agent id should not be equal zero which is the convention for empty cell."

class ColumnIsFullError(Exception):
    def __init__(self, col_index):
        self.message = "You cannot play in column {} because it is full.".format(col_index)
