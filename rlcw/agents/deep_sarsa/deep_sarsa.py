"""
author: Helen
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.deep_sarsa.networks import StateActionNetwork
import agents.common.utils as agent_utils
from agents.common.policy import EpsilonGreedyPolicy
from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer


class DeepSarsaAgent(CheckpointAgent):
    """
        Deep Sarsa agent to solve Lunar lander

        This was inspired by: https://github.com/JohDonald/Deep-Q-Learning-Deep-SARSA-LunarLander-v2
    """

    def name(self):
        return "deep_sarsa"

    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]

        self.criterion = nn.MSELoss()

        self.network = None
        self.optim = None
        self.policy = None

    def assign_env_dependent_variables(self, action_space, state_space):
        state_space = state_space.shape[0]
        action_space = action_space.n

        self.network, self.optim = agent_utils.with_optim(StateActionNetwork(state_space, action_space),
                                                          lr=self.learning_rate)

        self.policy = EpsilonGreedyPolicy(self.epsilon, action_space, self.device)

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        network_output_to_numpy = self.network(state)
        return self.policy.get_action(network_output_to_numpy)

    def train(self, training_context: ReplayBuffer):
        states, next_states, actions, next_actions, rewards, terminals = training_context.random_sample_sarsa(
            self.batch_size)
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        actions = torch.Tensor(actions)
        next_actions = torch.Tensor(next_actions)
        rewards = torch.Tensor(rewards)
        terminals = torch.Tensor(terminals)

        self.update_network(states, next_states, actions, next_actions, rewards, terminals)

    def update_network(self, state, next_state, action, next_action, reward, terminals):

        q_action = torch.gather(self.network(state), dim=1, index=action.long())

        q_next_action = torch.gather(self.network(next_state), dim=1, index=next_action.long())

        qsa_next_target = reward + (self.gamma * q_next_action) * (1 - terminals)
        q_network_loss = self.criterion(q_action, qsa_next_target.detach())
        self.optim.zero_grad()
        q_network_loss.backward()
        self.optim.step()

    def save(self):
        self.save_checkpoint(self.network, "StateActionNetwork")

    def load(self, path):
        self.load_checkpoint(self.network, path, "StateActionNetwork")

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.995  # decays epsilon

        if self.epsilon <= 0.1:
            self.epsilon = 0.1
