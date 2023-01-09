"""
author: Helen
"""
import numpy as np

from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer


class SarsaAgent(AbstractAgent):
    """
        Sarsa agent to solve LunarLander
    """
    def name(self):
        return "sarsa"

    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]

        self.Q = None

    def assign_env_dependent_variables(self, action_space, state_space):
        self.action_space = action_space
        self.Q = self._make_q()

    def _make_q(self):

        n_states = np.ones(8).shape * np.array([5, 5, 2, 2, 2, 2, 0, 0])
        n_states = np.round(n_states, 0).astype(int) + 1

        n_actions = self.action_space.n

        return np.zeros(
            [n_states[0], n_states[1], n_states[2], n_states[3], n_states[4], n_states[5], n_states[6], n_states[7],
             n_actions])

    @staticmethod
    def _continuous_to_discrete(observation):
        min_obs = observation.min()
        discrete = (observation - min_obs) * np.array([5, 5, 2, 2, 2, 2, 0, 0])
        return np.round(discrete, 0).astype(int)

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.995  # decays epsilon

        if self.epsilon <= 0.1:
            self.epsilon = 0.1

    def get_action(self, observation):
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            state_discrete = self._continuous_to_discrete(observation)
            action = np.argmax(self.Q[state_discrete[0], state_discrete[1], state_discrete[2], state_discrete[3],
                                      state_discrete[4], state_discrete[5], state_discrete[6], state_discrete[7]])
        return action

    def train(self, training_context: ReplayBuffer):
        states, next_states, actions, next_actions, rewards, terminals = training_context.random_sample_sarsa(
            self.batch_size)
        for i in range(0, self.batch_size):
            s = self._continuous_to_discrete(states[i])
            ns = self._continuous_to_discrete(next_states[i])
            na = int(next_actions[i][0])

            delta = self.learning_rate * (
                    rewards[i]
                    + self.gamma * self.Q[ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], na]
                    - self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], int(actions[i][0])]
            )

            self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], int(actions[i][0])] += delta
