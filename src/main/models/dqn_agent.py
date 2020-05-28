import datetime
import matplotlib.pyplot as plt

from .agent import _Agent
from .replay import Replay
from .nets import *
from ..environment.errors import ColumnIsFullError


class DQNAgent(_Agent):

    def __init__(self,
                 env,
                 net_name='CFConv2',
                 n_players=2,
                 encoding='3d',
                 epsilon=0.25,
                 discount=0.95,
                 num_episodes=1000,
                 batch_size=200,
                 num_replays=1000,
                 save_dir="../../models",
                 model_name="",
                 target_model_update_freq=10,
                 load_dir=""
                 ):
        super().__init__()
        self.action_space_size = env.get_action_space_size()
        self._net = self.init_net(net_name,
                                 env.get_shape(),
                                 env.get_action_space_size(),
                                 True,
                                 encoding,
                                 n_players,
                                 load_dir)
        self.epsilon = epsilon
        self.discount = discount
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_replays = num_replays
        self.replays = []
        self.target_model_update_freq = target_model_update_freq
        self.save_dir = save_dir
        self.model_name = self.get_model_name(model_name)

    @property
    def net(self):
        return self._net

    @net.setter
    def net(self, load_dir):
        self._net = load_net(load_dir)


    @staticmethod
    def init_net(net_name, env_shape, action_space_size, trainable, encoding, n_players, load_dir):
        if not load_dir:
            net_class = getattr(sys.modules[__name__], net_name)
            return net_class(action_space_size, env_shape, trainable, encoding, n_players)
        else:
            return load_net(load_dir)

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
            current_player_id = env.get_current_player_id()
            action = np.random.choice(self.action_space_size)
            try:
                reward, new_state = env.apply_action(action)
            except ColumnIsFullError:
                continue
            replay = Replay(prior_state, action, reward, new_state, current_player_id)
            replays.append(replay)
            game_is_finished = env.is_terminal_state(env.get_state())
            if env.is_blocked():
                game_is_finished = True
        return replays

    def play_action(self, env):
        action_id = self.select_action(env)
        reward, new_state = env.apply_action(action_id)
        return reward, new_state

    def select_action(self, env, seed=None):
        if seed is not None:
            np.random.seed(seed)
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
        # Initialize target net to reduce bias during training
        target_model = self.net.init_model()
        target_model.set_weights(self.net.model.get_weights())
        for episode in range(self.num_episodes):
            print("----------- Train on episode %i/%i (%s)" % (episode+1,
                                                               self.num_episodes,
                                                               datetime.datetime.now().strftime("%Hh%Mmn%Ss-%d/%m/%Y")))
            game_is_finished = False
            while not game_is_finished:
                prior_state = env.get_state()
                player_id = env.current_token_id
                actions = self.net.model.predict(self.net.process_input(np.expand_dims(prior_state, axis=0),
                                                                        self.net.encoding,
                                                                        self.net.n_players)).ravel()
                action = self.epsilon_greedy_predict_action(actions)
                try:
                    reward, new_state = env.apply_action(action)
                    episode_reward += reward
                    replay = Replay(prior_state, action, reward, new_state, player_id)
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
                    batch_post_states_q_values = target_model.predict(self.net.process_input(batch_post_states,
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
            if episode % self.target_model_update_freq == 0:
                print("Update target net")
                target_model.set_weights(self.net.model.get_weights())
            env.reset()
            total_rewards_per_episode[episode] = episode_reward
            episode_reward = 0
        print("Training done!")
        self.save_training(total_rewards_per_episode)

    def save_training(self, total_rewards_per_episode):
        os.makedirs(os.path.join(self.save_dir, self.model_name, 'net'), exist_ok=True)
        self.net.save(os.path.join(self.save_dir, self.model_name, 'net'))
        self.save_training_figures(total_rewards_per_episode)
        print("Training outputs saved in {}".format(os.path.abspath(os.path.join(self.save_dir, self.model_name))))

    def save_training_figures(self, rewards, figsize=(15, 8), extension="pdf"):
        plt.figure(figsize=figsize)
        plt.plot(rewards, color='r')
        plt.grid()
        plt.title("Reward per training episode")
        plt.ylabel("Rewards")
        plt.xlabel("Episodes")
        plt.xticks(range(1, len(rewards)))
        plt.xlim(0, len(rewards)-1)
        os.makedirs(os.path.join(self.save_dir, self.model_name, 'figures'))
        plt.savefig(os.path.join(self.save_dir, self.model_name, 'figures', "rewards_per_episode.%s" % extension))
        with open(os.path.join(self.save_dir, self.model_name, 'figures', "rewards_per_episode.txt"), "a") as f:
            for episode_reward in rewards:
                f.write(str(episode_reward))

    def save_replay(self, replay):
        self.replays.pop(0)
        self.replays.append(replay)

    def sample_batch(self):
        batch = np.random.choice(self.replays, self.batch_size, replace=False)
        # Toggle ids if necessary so it always looks the current player id is 2
        processed_batch = np.array([r.toggle_ids() if r._current_player_id == 1 else r for r in batch])
        return processed_batch

    def epsilon_greedy_predict_action(self, actions):
        if np.random.random_sample() < self.epsilon:
            return np.random.randint(0, len(actions))
        else:
            return np.argmax(actions)

    @staticmethod
    def get_model_name(model_name):
        if not isinstance(model_name, str):
            raise TypeError("model_name must have string type")

        if model_name == "":
            date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
            return "trained_model_" + date
        return model_name
