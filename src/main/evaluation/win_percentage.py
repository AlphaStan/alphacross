import numpy as np

from ..environment.errors import ColumnIsFullError


class RandomAgentEvaluator:

    def __init__(self, agent, environment, num_episodes=100):
        self.agent = agent
        self.environment = environment
        self.num_episodes = num_episodes

    def _make_one_move(self, action, player_id):
        has_played = False
        has_won = False
        is_blocked = False
        try:
            _, new_state = self.environment.apply_action(action)
            has_played = True
            if self.environment._is_winning_move(new_state, action, player_id):
                has_won = True
            elif self.environment.is_blocked():
                is_blocked = True
        except ColumnIsFullError:
            pass
        return has_played, has_won, is_blocked

    def evaluate(self):
        n_actions = self.environment.get_action_space_size()
        random_agent_id = 1
        agent_id = 2
        percentages = {'agent_winning_percentage': 0,
                       'random_agent_winning_percentage': 0,
                       'draw_percentage': 0}
        for episode_index in range(self.num_episodes):
            self.environment.reset()
            episode_is_finished = False
            while not episode_is_finished:
                random_agent_has_played = False
                agent_has_played = False
                while not random_agent_has_played:
                    random_agent_action = np.random.randint(n_actions)
                    random_agent_has_played, random_agent_has_won, environment_is_blocked = self._make_one_move(
                        random_agent_action,
                        random_agent_id)
                    if random_agent_has_won:
                        percentages['random_agent_winning_percentage'] += 1
                        episode_is_finished = True
                    elif environment_is_blocked:
                        percentages['draw_percentage'] += 1
                        episode_is_finished = True
                while not agent_has_played and not episode_is_finished:
                    agent_action = self.agent.select_action(self.environment)
                    agent_has_played, agent_has_won, environment_is_blocked = self._make_one_move(
                        agent_action,
                        agent_id)
                    if agent_has_won:
                        percentages['agent_winning_percentage'] += 1
                        episode_is_finished = True
                    elif environment_is_blocked:
                        percentages['draw_percentage'] += 1
                        episode_is_finished = True
        for key in percentages:
            percentages[key] /= self.num_episodes
        return percentages
