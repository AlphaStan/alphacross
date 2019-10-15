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
        one_player_has_won = False
        number_of_rounds = 0
        number_of_cells = self.get_nb_columns() * self.get_nb_rows()
        while number_of_rounds < number_of_cells:

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
                except OutOfGridError:
                    print("Player {}, you should give a number between 0 and 6.".format(agent_id))
                except ColumnIsFullError:
                    print("Player {}, column {} is full, please select another number between 0 and 6." \
                          .format(agent_id, column_id))

            self.display()
            if self.is_winning_move(column_id, agent_id):
                print("Congratulation player {}, you have won !".format(agent_id))
                break

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

    def is_winning_move(self, col_index, agent_id):
        return (self.check_vertical_victory(col_index, agent_id) or \
                self.check_horizontal_victory(col_index, agent_id) or \
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
                left_border = max(0, col_index - 3)
                right_border = min(7, col_index + 3)
                row_index = 6 - reversed_row_id - 1
                bottom_border = max(0, row_index - 3)
                top_border = min(5, row_index + 3)
                ascending_diagonal = [self._grid[col_id][row_id] for col_id, row_id
                                      in zip(range(left_border, right_border), range(bottom_border, top_border + 1))]
                descending_diagonal = [self._grid[col_id][row_id] for col_id, row_id
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

    def display(self):
        print(self.convert_grid_to_string())
        print()
