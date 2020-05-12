import datetime
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.models import load_model

from .agent import _Agent
from .loss import dqn_mask_loss
from .replay import Replay
from ..environment.errors import ColumnIsFullError


#TODO: test class for that
class DQNAgent(_Agent):

    def __init__(self,
                 env,
                 epsilon=0.25,
                 discount=0.95,
                 num_episodes=1000,
                 batch_size=200,
                 num_replays=1000,
                 save_dir="../../models"
                 ):
        super().__init__()
        self.action_space_size = env.get_action_space_size()
        self.model = self.init_model(env.get_shape(), env.get_state_space_size(), env.get_action_space_size())
        self.epsilon = epsilon
        self.discount = discount
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_replays = num_replays
        self.replays = []
        self.save_dir = save_dir

    @staticmethod
    def init_model(env_shape, state_space_size, action_space_size):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=env_shape))
        model.add(tf.keras.layers.Dense(24, activation=tf.keras.activations.relu, input_dim=state_space_size))
        model.add(tf.keras.layers.Dense(action_space_size, activation=tf.keras.activations.softmax))
        model.compile(loss=dqn_mask_loss, optimizer='Adam', metrics=['accuracy'])
        return model

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

    def select_action(self, env):
        state = np.expand_dims(env.get_state(), axis=0)
        action_probabilities = self.model.predict(state)
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
                player_id = env.current_token_id
                actions = self.model.predict(np.expand_dims(prior_state, axis=0)).ravel()
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
                    batch_post_states_q_values = self.model.predict(batch_post_states)
                    batch_prior_states_q_values = batch_rewards + self.discount * batch_post_states_q_values.max(axis=1)
                    batch_actions = np.array([replay._action for replay in batch])
                    batch_targets = np.concatenate([np.expand_dims(batch_prior_states_q_values, axis=1),
                                                    np.expand_dims(batch_actions, axis=1)],
                                                   axis=1)
                    self.model.train_on_batch(batch_prior_states, batch_targets)

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
        self.model.save(os.path.join(self.save_dir, "trained_model_%s" % date, 'trained_model.h5'))
        self.save_training_figures(total_rewards_per_episode, date)
        print("Training outputs saved in {}".format(os.path.abspath(os.path.join(self.save_dir,
                                                                                 "trained_model_%s/" % date))))

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
