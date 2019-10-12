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

    def play(self):
        player1_has_won = False
        player2_has_won = False
        number_of_rounds = 0
        while not player1_has_won or not player2_has_won or number_of_rounds < 42:

            if number_of_rounds % 2 == 0:
                agent_id = 1
            if number_of_rounds % 2 != 0:
                agent_id = 2

            agent_has_played = False
            while not agent_has_played:
                try:
                    column_id = input("Player {}, please give the column number where you play".format(agent_id))
                    self.put_token(column_id, agent_id)
                    agent_has_played = True
                    break
                except OutOfGridError:
                    print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                    break
                except ColumnIsFullError:
                    print("Player {}, column {} is full, please select another number between 0 and 6." \
                          .format(agent_id, column_id))
                    break

            if player1_has_won:
                print("Congratulations player 1 !")
                return

            if player2_has_won:
                print("Congratulations player 2 !")
                return
            number_of_rounds += 1
            self._last_player_agent_id = agent_id

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

    def convert_grid_to_string(self):
        rows_list = ["|", "|", "|", "|", "|", "|"]
        for column in self._grid:
            for i in range(len(column)):
                rows_list[i] += str(column[i]) if column[i] else " "
                rows_list[i] += "|"
        rows = "\n".join(rows_list[::-1])
        return rows

    def display(self):
        print(self._convert_grid_to_string())
