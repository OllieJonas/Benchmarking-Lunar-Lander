from typing import NoReturn

import numpy as np
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from abstract_agent import CheckpointedAbstractAgent
from replay_buffer import ReplayBuffer


class SimpleNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, no_actions):
        super(SimpleNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, no_actions),
        )

    def forward(self, x):
        return self.layers(x)


class DeepQNetwork(CheckpointedAbstractAgent):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)
        self.value_network = SimpleNetwork()

        self.target_value_network = None
        self._sync_target_network()

        self.optimiser = optim.Adam()
        self.loss = nn.HuberLoss()

    def _sync_target_network(self):
        self.target_value_network = deepcopy(self.value_network)

    def save(self):
        pass

    def load(self):
        pass

    def name(self) -> str:
        return "DQN"

    def get_action(self, state):
        pass

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        pass
