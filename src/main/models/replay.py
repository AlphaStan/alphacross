class Replay:

    TOGGLE_TABLE = {0: 0, 1: 2, 2: 1}

    def __init__(self, prior_state, action, reward, post_state, current_player_id):
        self._prior_state = prior_state
        self._action = action
        self._reward = reward
        self._post_state = post_state
        self._current_player_id = current_player_id

    def toggle_ids(self):
        self._prior_state = self.toggle_state(self._prior_state)
        self._post_state = self.toggle_state(self._post_state)
        return self

    @classmethod
    def toggle_state(cls, state):
        return [[cls.TOGGLE_TABLE[token_id] for token_id in column] for column in state]
