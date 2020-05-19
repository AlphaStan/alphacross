from tensorflow.python.keras.models import load_model
import datetime
import os
import matplotlib.pyplot as plt
import logging
import sys

from .agent import _Agent
from ..environment.errors import ColumnIsFullError
from .nets import *


#TODO: test class for that
class DQNAgent(_Agent):

    def __init__(self,
                 env,
                 net_name='CFConv2',
                 n_players=2,
                 encoding='2d',
                 epsilon=0.25,
                 discount=0.95,
                 num_episodes=1000,
                 batch_size=200,
                 num_replays=1000,
                 save_dir="../../models"
                 ):
        super().__init__()
        self.action_space_size = env.get_action_space_size()
        self.net = self.init_model(net_name, env.get_shape(), env.get_action_space_size(), True, encoding, n_players)
        self.epsilon = epsilon
        self.discount = discount
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_replays = num_replays
        self.replays = []
        self.save_dir = save_dir

    @staticmethod
    def init_model(net_name, env_shape, action_space_size, trainable, encoding, n_players):
        net_class = getattr(sys.modules[__name__], net_name)
        return net_class(action_space_size, env_shape, trainable, encoding, n_players)

    def init_replays(self, env):
        replays = []
        while len(replays) < self.num_replays:
            replays += self.generate_replay(env)
            env.reset()
            print("Replays generated %i/%i" % (min(len(replays), self.num_replays), self.num_replays))
        self.replays = replays[:self.num_replays]

    def generate_replay(self, env):
        replays = []
        game_is_finished = False
        while not game_is_finished:
            prior_state = env.get_state()
            action = np.random.choice(self.action_space_size)
            try:
                reward, new_state = env.apply_action(action)
            except ColumnIsFullError:
                continue
            replay = Replay(prior_state, action, reward, new_state)
            replays.append(replay)
            game_is_finished = env.is_terminal_state(env.get_state())
            if env.is_blocked():
                game_is_finished = True
        return replays

    def play_action(self, env):
        action_id = self.select_action(env)
        reward, new_state = env.apply_action(action_id)
        return reward, new_state

    def select_action(self, env):
        state = self.net.process_input(np.expand_dims(env.get_state(), axis=0),
                                       self.net.encoding,
                                       self.net.n_players)
        action_probabilities = self.net.model.predict(state)
        action_id = self.epsilon_greedy_predict_action(action_probabilities)
        return action_id

    def train(self, env):
        if not self.replays:
            raise AttributeError("Attempting to train DQNAgent with no replays, use generate_replays first")
        total_rewards_per_episode = np.zeros(self.num_episodes)
        episode_reward = 0
        for episode in range(self.num_episodes):
            print("----------- Train on episode %i/%i (%s)" % (episode+1,
                                                               self.num_episodes,
                                                               datetime.datetime.now().strftime("%Hh%Mmn%Ss-%d/%m/%Y")))
            game_is_finished = False
            while not game_is_finished:
                prior_state = env.get_state()
                actions = self.net.model.predict(self.net.process_input(np.expand_dims(prior_state, axis=0),
                                                                        self.net.encoding,
                                                                        self.net.n_players)).ravel()
                action = self.epsilon_greedy_predict_action(actions)
                try:
                    reward, new_state = env.apply_action(action)
                    episode_reward += reward
                    replay = Replay(prior_state, action, reward, new_state)
                    self.save_replay(replay)

                    batch = self.sample_batch()
                    batch_prior_states = np.concatenate(
                        [np.expand_dims(replay._prior_state, axis=0) for replay in batch],
                        axis=0)
                    batch_post_states = np.concatenate(
                        [np.expand_dims(replay._post_state, axis=0) for replay in batch],
                        axis=0)
                    batch_rewards = np.array([replay._reward for replay in batch])

                    # A quick explanation here:
                    # We want to optimise the network so it approximates the optimal Q function
                    # The Q function is defined by the Bellman optimality equation:
                    # Q(s,a) = r + max_a Q(s', a)
                    # The difference between the two sides is the temporal difference:
                    # delta = Q(s, a) - r - max_a Q(s', a)
                    # This is the quantity we want to perform the gradient descent on
                    # In this block the target is defined as the right hand side of the Bellman
                    # equation, the network is used as an approximation of the Q function to define this target
                    batch_post_states_q_values = self.net.model.predict(self.net.process_input(batch_post_states,
                                                                                               self.net.encoding,
                                                                                               self.net.n_players))
                    batch_prior_states_q_values = batch_rewards + self.discount * batch_post_states_q_values.max(axis=1)
                    batch_actions = np.array([replay._action for replay in batch])
                    batch_targets = np.concatenate([np.expand_dims(batch_prior_states_q_values, axis=1),
                                                    np.expand_dims(batch_actions, axis=1)],
                                                   axis=1)
                    self.net.model.train_on_batch(self.net.process_input(batch_prior_states,
                                                                         self.net.encoding,
                                                                         self.net.n_players),
                                                  batch_targets)

                    game_is_finished = env.is_terminal_state(env.get_state())
                    if env.is_blocked():
                        game_is_finished = True
                except ColumnIsFullError:
                    continue
            env.reset()
            total_rewards_per_episode[episode] = episode_reward
            episode_reward = 0
        print("Training done!")
        date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        os.mkdir(os.path.join(self.save_dir, "trained_model_%s" % date))
        self.net.model.save(os.path.join(self.save_dir, "trained_model_%s" % date, 'trained_model.h5'))
        self.save_training_figures(total_rewards_per_episode, date)
        print("Training outputs saved in {}".format(os.path.abspath(os.path.join(self.save_dir,
                                                                                 "trained_model_%s/" % date))))

    def compare_two_models(self, first_model_path, second_model_path, env, n_episodes):
        first_model = load_model(first_model_path)
        second_model = load_model(second_model_path)

        n_first_model_wins = 0
        n_second_model_wins = 0
        n_draws = 0

        def make_one_move(model):
            model_made_one_move = False
            current_state = env.get_state()
            model_has_won = False
            state_is_terminal = False
            while not model_made_one_move:
                model_actions = model(np.expand_dims(current_state, axis=0)).ravel()
                model_action = self.epsilon_greedy_predict_action(model_actions)
                try:
                    _, _ = env.apply_action(model_action)
                    model_has_won = env.is_terminal_state(env.get_state())
                    if env.is_blocked():
                        state_is_terminal = True
                except ColumnIsFullError:
                    continue
                model_made_one_move = True
            return env, model_has_won, state_is_terminal

        # The model plays against itself
        game_is_finished = False
        for episode in range(n_episodes):
            while not game_is_finished:
                env, first_model_won, state_is_terminal = make_one_move(first_model)
                if first_model_won:
                    n_first_model_wins += 1
                    game_is_finished = True
                elif state_is_terminal:
                    n_draws += 1
                    game_is_finished = True
                else:
                    env, second_model_won, state_is_terminal = make_one_move(second_model)
                    if second_model_won:
                        n_second_model_wins += 1
                        game_is_finished = True
                    elif state_is_terminal:
                        n_draws += 1
                        game_is_finished = True
            env.reset()
        if n_first_model_wins > n_second_model_wins:
            winner_message = "{} model performs best ({})".format("First", first_model_path)
        elif n_first_model_wins < n_second_model_wins:
            winner_message = "{} model performs best ({})".format("Second", second_model_path)
        else:
            winner_message = "It is a draw"
        message = """
        Number of episodes {} \n
        First model won {} times \n
        Second model won {} times \n
        Draws {} \n
        Winner message {}
        """.format(
            n_episodes,
            n_first_model_wins,
            n_second_model_wins,
            n_draws,
            winner_message
        )
        logging.info(message)

    def evaluate(self, first_model_path, second_model_path, env, n_episodes):

        def make_one_move():
            model_made_one_move = False
            current_state = env.get_state()
            reward = 0
            state_is_terminal = False
            while not model_made_one_move:
                model_actions = self.net.model(np.expand_dims(current_state, axis=0)).ravel()
                model_action = self.epsilon_greedy_predict_action(model_actions)
                try:
                    reward, _ = env.apply_action(model_action)
                    state_is_terminal = env.is_terminal_state(env.get_state())
                    if env.is_blocked():
                        state_is_terminal = True
                except ColumnIsFullError:
                    continue
                model_made_one_move = True
            return env, reward, state_is_terminal

        def make_one_random_move():
            model_made_one_move = False
            reward = 0
            state_is_terminal = False
            while not model_made_one_move:
                model_actions = np.ones((env.get_action_space_size(), 1)) / env.get_action_space_size()
                model_action = self.epsilon_greedy_predict_action(model_actions)
                try:
                    reward, _ = env.apply_action(model_action)
                    state_is_terminal = env.is_terminal_state(env.get_state())
                    if env.is_blocked():
                        state_is_terminal = True
                except ColumnIsFullError:
                    continue
                model_made_one_move = True
            return env, reward, state_is_terminal

        # The model plays against itself
        game_is_finished = False
        model_total_reward_per_episode = np.zeros(n_episodes)
        model_episode_reward = 0
        for episode in range(n_episodes):
            while not game_is_finished:
                env, model_reward, game_is_finished = make_one_move()
                model_episode_reward += model_reward
                if not game_is_finished:
                    # There might be a problem here
                    # The model learnt to be player 2, thus it is potentially biased
                    # It might be necessary to switch the ids in the state
                    # However I am not sure about it, the model learnt from samples in the replays
                    # It may be ok, to be checked
                    # Actually, in this context we don't really care, the point is just to simulate another player
                    env, _, game_is_finished = make_one_random_move()
            env.reset()
            model_total_reward_per_episode[episode] = model_episode_reward
        average_reward_per_episode = np.mean(model_total_reward_per_episode)
        return average_reward_per_episode

    def save_training_figures(self, rewards, date, figsize=(15, 8), extension="pdf"):
        plt.figure(figsize=figsize)
        plt.plot(rewards, color='r')
        plt.grid()
        plt.title("Reward per training episode")
        plt.ylabel("Rewards")
        plt.xlabel("Episodes")
        plt.xticks(range(1, len(rewards)))
        plt.xlim(0, len(rewards)-1)
        plt.savefig(os.path.join(self.save_dir, "trained_model_%s" % date, "rewards_per_episode.%s" % extension))
        with open(os.path.join(self.save_dir, "trained_model_%s" % date, "rewards_per_episode.txt"), "a") as f:
            for episode_reward in rewards:
                f.write(str(episode_reward))

    def get_legal_action(self, state, get_action_prob, env):
        action_probabilities = get_action_prob(state)
        legal_actions = [env.is_legal_action(state, action) for action in range(action_probabilities)]
        legal_action_prob = [prob if legal_action else 0
                             for prob, legal_action in zip(action_probabilities, legal_actions)]

        if sum(legal_action_prob) == 0:
            return legal_action_prob
        return legal_action_prob / sum(legal_action_prob)

    def save_replay(self, replay):
        self.replays.pop(0)
        self.replays.append(replay)

    def sample_batch(self):
        return np.random.choice(self.replays, self.batch_size, replace=False)

    # DEPRECATED
    def get_mini_batch_targets(self, mini_batch):
        warnings.warn("'get_mini_batch_target' is deprecated and should not be used as is")
        return np.array([replay._reward if replay._reward == 1
                         else self.discount * np.max(self.net.model.predict(np.expand_dims(replay._post_state, axis=0)))
                         for replay in mini_batch])

    def epsilon_greedy_predict_action(self, actions):
        if np.random.random_sample() < self.epsilon:
            return np.random.randint(0, len(actions))
        else:
            return np.argmax(actions)


class Replay:
    def __init__(self, prior_state, action, reward, post_state):
        self._prior_state = prior_state
        self._action = action
        self._reward = reward
        self._post_state = post_state
