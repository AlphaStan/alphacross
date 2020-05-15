import numpy as np

from ..environment.errors import ColumnIsFullError


class RandomAgentEvaluator:
    """
    Return the percentage of victories over several games against a random agent
    """

    def __init__(self, agent, environment, num_episodes=100):
        self.agent = agent
        self.environment = environment
        self.num_episodes = num_episodes
        self.random_agent_id = 1
        self.agent_id = 2
        self._percentages = self._init_percentages()

    @staticmethod
    def _init_percentages():
        return {'agent_winning_percentage': 0, 'random_agent_winning_percentage': 0, 'draw_percentage': 0}

    def _make_one_move(self, action, player_id):
        has_played = False
        episode_is_finished = False
        try:
            _, new_state = self.environment.apply_action(action)
            has_played = True
            if self.environment._is_winning_move(new_state, action, player_id):
                if player_id == 1:
                    self._percentages['random_agent_winning_percentage'] += 1. / self.num_episodes
                else:
                    self._percentages['agent_winning_percentage'] += 1. / self.num_episodes
                episode_is_finished = True
            elif self.environment.is_blocked():
                self._percentages['draw_percentage'] += 1. / self.num_episodes
                episode_is_finished = True
        except ColumnIsFullError:
            pass
        return has_played, episode_is_finished

    def evaluate(self):
        n_actions = self.environment.get_action_space_size()
        self._percentages = self._init_percentages()
        for episode_index in range(self.num_episodes):
            self.environment.reset()
            episode_is_finished = False
            while not episode_is_finished:
                random_agent_has_played = False
                agent_has_played = False
                while not random_agent_has_played:
                    random_agent_action = np.random.randint(n_actions)
                    random_agent_has_played, episode_is_finished = self._make_one_move(random_agent_action,
                                                                                       self.random_agent_id)
                while not agent_has_played and not episode_is_finished:
                    agent_action = self.agent.select_action(self.environment)
                    agent_has_played, episode_is_finished = self._make_one_move(agent_action,
                                                                                self.agent_id)
        return self._percentages
