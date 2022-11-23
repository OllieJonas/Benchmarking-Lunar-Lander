import random

import numpy as np

from typing import NoReturn, List

from agents.abstract_agent import AbstractAgent


class SarsaAgent(AbstractAgent):

    def __init__(self, logger, action_space, observation_space, config):
        self.observation_space =  observation_space
        self.action_space = action_space

        self.q_table = self._init_q_table()
        self.learn = 0.5
        self.gamma = 0.9
        self.epsilon = 0.5

        super().__init__(logger, action_space, config)


    def _init_q_table(self) -> np.array:
            high = self.observation_space.high
            low = self.observation_space.low
            
            n_states = (high - low)[0:6] * np.array([10, 10, 5, 5, 1, 1])
            n_states = np.round(n_states, 0).astype(int) + 1

            return np.zeros([n_states[0], n_states[1], n_states[2], n_states[3], n_states[4], n_states[5], self.action_space.n])

    def name(self):
        return "sarsa"
        
    def _continuous_to_discrete(self, state):
        min_states = self.observation_space.low
        state_discrete = (state[0] - min_states)[0:6] * np.array([10, 10, 5, 5, 1, 1])
        return np.round(state_discrete, 0).astype(int)
    

    def get_action(self, state):
        epsilon = self.epsilon
        if epsilon and random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()

        else:
            state_discrete = self._continuous_to_discrete(state)
            action = np.argmax(self.q_table[state_discrete[0], state_discrete[1], state_discrete[2], state_discrete[3], state_discrete[4], state_discrete[5]])
        
        return action
        
    def train(self, training_context: List) -> NoReturn:
        s = self._continuous_to_discrete(training_context[0][0])
        ns = self._continuous_to_discrete(training_context[0][1])
        reward = training_context[0][2]
        a = training_context[0][3]
        na = self.get_action(training_context[1])

        delta = self.learn * (
                reward
                + self.gamma * self.q_table[ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], na]
                - self.q_table[s[0], s[1], s[2], s[3], s[4], s[5], a]
        )

        self.q_table[s[0], s[1], s[2], s[3], s[4], s[5], a] += delta
