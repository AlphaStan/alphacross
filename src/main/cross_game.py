
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

    def _display_grid(self):
        rows_list = ["|", "|", "|", "|", "|", "|"]
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def display(self):
        print(self._display_grid())


class ZeroAgentIdError(Exception):
    def __init__(self):
        self.message = "Agent id should not be equal zero which is the convention for empty cell."

class ColumnIsFullError(Exception):
    def __init__(self, col_index):
        self.message = "You cannot play in column {} because it is full.".format(col_index)