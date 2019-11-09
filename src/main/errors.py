class ZeroAgentIdError(Exception):
    def __init__(self):
        self.message = "Agent id should not be equal to zero which is the convention for empty cell."
        Exception.__init__(self, self.message)


class ColumnIsFullError(Exception):
    def __init__(self, col_index):
        self.message = "You cannot play in column {} because it is full.".format(col_index)
        Exception.__init__(self, self.message)


class OutOfGridError(Exception):
    def __init__(self, agent_id, col_index, nb_cols):
        self.message = "Player {} has selected selected column {} but there are only {} columns"\
            .format(agent_id, col_index, nb_cols)
        Exception.__init__(self, self.message)


class AlreadyPlayedError(Exception):
    def __init__(self, agent_id):
        self.message = "Player {} has already played, cannot play twice in a row".format(agent_id)
        Exception.__init__(self, self.message)
