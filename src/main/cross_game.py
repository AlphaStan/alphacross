
class CrossGame:

    def __init__(self):
        self._NB_COLUMNS = 7
        self._NB_ROWS = 6
        self._init_grid()

    def _init_grid(self):
        self._grid = [[0 for _ in range(self._NB_ROWS)] for _ in range(self._NB_COLUMNS)]

    def put_token(self, col_index, agent_id):
        if agent_id == 0:
            raise ZeroAgentIdError()
        if col_index >= self._NB_COLUMNS:
            raise OutOfGridError(agent_id, col_index, self._NB_COLUMNS)
        if self._grid[col_index][self._NB_ROWS - 1] != 0:
            raise ColumnIsFullError(col_index)
        for i, slot in enumerate(self._grid[col_index]):
            if slot == 0:
                self._grid[col_index][i] = agent_id
                break

class ZeroAgentIdError(Exception):
    def __init__(self):
        self.message = "Agent id should not be equal zero which is the convention for empty cell."

class ColumnIsFullError(Exception):
    def __init__(self, col_index):
        self.message = "You cannot play in column {} because it is full.".format(col_index)

class OutOfGridError(Exception):
    def __init__(self, agent_id, col_index, nb_cols):
        self.message = "Player {} has selected selected column {} but there are only {} columns".format(agent_id,
                                                                                                        col_index,
                                                                                                        nb_cols)