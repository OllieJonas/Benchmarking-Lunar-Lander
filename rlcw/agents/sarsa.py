import random

import numpy as np

from typing import NoReturn, List

from agents.abstract_agent import AbstractAgent


class SarsaAgent(AbstractAgent):

    def __init__(self, action_space):
        self.action_space = action_space
        self.Q = self._make_q(np.zeros(8))
        self.alpha = 0.1 
        self.gamma = 0.9 

        
        super().__init__(action_space)

    def name(self):
        return "sarsa"

    def _make_q(self, observation):

        n_states = (np.ones(8).shape) * np.array([100, 1000])
        n_states = np.round(n_states, 0).astype(int) + 1

        n_actions = self.action_space.n 
        
        return np.zeros([n_states[0], n_states[1], n_actions])

        
    def _continuous_to_discrete(self, observation):
        
        min_obs = observation.min()
        discrete = (observation - min_obs).reshape(4, 2) * np.array([10,100])
        return np.round(discrete, 0).astype(int)

    def get_action(self, observation):
        epsilon = 0.9

        if np.random.uniform(0, 1) < epsilon:
            action = self.action_space.sample()
        else:
            state_discrete = self._continuous_to_discrete(observation)
            action = np.argmax(self.Q[state_discrete[0], state_discrete[1]])
        return action


    def train(self, training_context: List) -> NoReturn:
        #import pdb; pdb.set_trace()
        state = training_context[0]["curr_obsv"]
        state2 = training_context[0]["next_obsv"]
        action = training_context[0]["action"]
        reward = training_context[0]["reward"]


        s = self._continuous_to_discrete(state)
        ns = self._continuous_to_discrete(state2)
        na = self.get_action(state2)

        delta = self.alpha * (
                reward
                + self.gamma * self.Q[ns[0], ns[1], na]
                - self.Q[s[0], s[1], action]
        )
        self.Q[s[0], s[1], action] += delta