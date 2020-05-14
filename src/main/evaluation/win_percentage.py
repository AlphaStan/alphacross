import numpy as np

from ..environment.errors import ColumnIsFullError


def against_random_agent(agent, environment, num_episodes=100):
    environment.reset()
    n_actions = environment.get_action_space_size()
    random_agent_id = 1
    agent_id = 2
    n_random_agent_victory = 0
    n_model_victory = 0
    n_draw = 0
    for episode_index in range(num_episodes):
        print("Episode %s/%s" % (episode_index, num_episodes))
        episode_is_finished = False
        while not episode_is_finished:
            model_has_played = False
            random_agent_has_played = False
            while not random_agent_has_played:
                try:
                    random_agent_action = np.random.randint(n_actions)
                    _, new_state = environment.apply_action(random_agent_action)
                    random_agent_has_played = True
                    print("Random agent has played")
                    if environment._is_winning_move(new_state, random_agent_action, random_agent_id):
                        n_random_agent_victory += 1
                        episode_is_finished = True
                        print("Random agent has won")
                        break
                    elif environment.is_blocked():
                        n_draw += 1
                        episode_is_finished = True
                        print("Draw")
                        break
                except ColumnIsFullError:
                    pass
            while not model_has_played:
                try:
                    agent_action = agent.select_action(environment)
                    _, new_state = environment.apply_action(agent_action)
                    model_has_played = True
                    print("DQN agent has played")
                    if environment._is_winning_move(new_state, agent_action, agent_id):
                        n_model_victory += 1
                        episode_is_finished = True
                        print("DQN agent has won")
                        break
                    elif environment.is_blocked():
                        n_draw += 1
                        episode_is_finished = True
                        print("Draw")
                        break
                except ColumnIsFullError:
                    pass
        environment.reset()
    return {'agent_winning_percentage': n_model_victory / num_episodes,
            'random_agent_winning_percentage': n_random_agent_victory / num_episodes,
            'draw_percentage': n_draw / num_episodes}


