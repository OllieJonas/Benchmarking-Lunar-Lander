"""
author: Helen
"""

import numpy as np


from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer

class SarsaAgent(CheckpointAgent):
    """
        Sarsa agent to solve Lunar lander
    """

    def name(self):
        return "sarsa"

    def __init__(self, logger, config):
        super().__init__(logger, config)
        
        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]

    def assign_env_dependent_variables(self, action_space, state_space):
        state_space = state_space.shape[0]
        self.action_space = action_space

        self.Q = self._make_q(np.zeros(8))

    def _make_q(self, observation):

        n_states = (np.ones(8).shape) * np.array([5, 5, 2, 2, 2, 2, 0, 0])
        n_states = np.round(n_states, 0).astype(int) + 1

        n_actions = self.action_space.n 
        
        return np.zeros([n_states[0], n_states[1], n_states[2], n_states[3], n_states[4], n_states[5], n_states[6], n_states[7], n_actions])

        
    def _continuous_to_discrete(self, observation):
        
        min_obs = observation.min()
        #import pdb; pdb.set_trace();
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
            action = np.argmax(self.Q[state_discrete[0], state_discrete[1], state_discrete[2], state_discrete[3], state_discrete[4], state_discrete[5], state_discrete[6], state_discrete[7]])
        return action

    def save(self):
        '''
            do nothing
        '''
    

    def load(self, path):
        '''
            do nothing
        '''

    def train(self, training_context: ReplayBuffer):
        states, actions, rewards, next_states, terminal = training_context.random_sample(self.batch_size)

        s = self._continuous_to_discrete(states[0])
        ns = self._continuous_to_discrete(next_states[0])
        na = self.get_action(next_states[0])

        delta = self.learning_rate * (
                rewards[0]
                + self.gamma * self.Q[ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7], na]
                - self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], int(actions[0])]
        )

        self.Q[s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], int(actions[0])] += delta
