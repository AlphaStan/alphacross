import numpy as np

from ..environment.errors import ColumnIsFullError


def against_random_agent(agent, environment, num_episodes=100):
    environment.reset()
    n_actions = environment.get_action_space_size()
    random_agent_id = 1
    agent_id = 2
    percentages = {'agent_winning_percentage': 0,
                   'random_agent_winning_percentage': 0,
                   'draw_percentage': 0}
    for episode_index in range(num_episodes):
        episode_is_finished = False
        while not episode_is_finished:
            model_has_played = False
            random_agent_has_played = False
            while not random_agent_has_played:
                try:
                    random_agent_action = np.random.randint(n_actions)
                    _, new_state = environment.apply_action(random_agent_action)
                    random_agent_has_played = True
                    if environment._is_winning_move(new_state, random_agent_action, random_agent_id):
                        percentages['random_agent_winning_percentage'] += 1
                        episode_is_finished = True
                        break
                    elif environment.is_blocked():
                        percentages['draw_percentage'] += 1
                        episode_is_finished = True
                        print("Draw")
                        break
                except ColumnIsFullError:
                    pass
            while not model_has_played and not episode_is_finished:
                try:
                    agent_action = agent.select_action(environment)
                    _, new_state = environment.apply_action(agent_action)
                    model_has_played = True
                    if environment._is_winning_move(new_state, agent_action, agent_id):
                        percentages['agent_winning_percentage'] += 1
                        episode_is_finished = True
                        break
                    elif environment.is_blocked():
                        percentages['draw_percentage'] += 1
                        episode_is_finished = True
                        break
                except ColumnIsFullError:
                    pass
        environment.reset()
    for key in percentages:
        percentages[key] /= num_episodes
    return percentages


