from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy

from agents.abstract_agent import CheckpointedAbstractAgent
from agents.dqn.policy import EpsilonGreedyPolicy
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

    def __init__(self, logger, action_space, observation_space, config):
        super().__init__(logger, action_space, config)
        # config vars

        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.sample_size = config["sample_size"]
        self.hidden_size = config["hidden_layer_size"]
        self.update_count = config["update_count"]

        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]

        self.no_actions = action_space.n

        self.value_network = SimpleNetwork(observation_space.shape[0], self.hidden_size, self.no_actions)\
            .to(self.device)

        self.target_value_network = None
        self._sync_target_network()

        self.value_network_optimiser = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.criterion = nn.HuberLoss()

        self._batch_cnt = 0
        self._update_cnt = 0

        self.policy = EpsilonGreedyPolicy(self.epsilon, self.no_actions)

    def _sync_target_network(self):
        self.target_value_network = deepcopy(self.value_network)

    def save(self):
        pass

    def load(self):
        pass

    def name(self) -> str:
        return "DQN"

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_values = self.value_network.forward(state)

        return self.policy.get_action(state)

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if self._batch_cnt < self.batch_size:
            self._batch_cnt += 1
        else:
            self._do_train(training_context)
            self._batch_cnt = 0
        pass

    def _do_train(self, training_context):
        states, next_states, rewards, actions, dones = \
            training_context.random_sample_transformed(self.sample_size, self.device)

        predicted = self.value_network.forward(states).gather(actions, 1)

        self.value_network.train()
        self.target_value_network.eval()

        with torch.no_grad():
            next_values = self.target_value_network.forward(next_states).detach().max(1)[0].unsqueeze(1)

        labels = rewards + (self.gamma * next_values * (1 - dones))

        loss = self.criterion(predicted, labels).to(self.device)
        self.value_network_optimiser.zero_grad()
        loss.backward()
        self.value_network_optimiser.step()

        self._soft_copy(self.value_network, self.target_value_network, self.tau)

        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

        if self._update_cnt >= self.update_count:
            self._sync_target_network()
            self._update_cnt = 0
        else:
            self._update_cnt += 1

    @staticmethod
    def _soft_copy(v, t_v, tau):
        value_state_dict = dict(v.named_parameters())
        target_value_state_dict = dict(t_v.named_parameters())

        for key in value_state_dict:
            value_state_dict[key] = tau * value_state_dict[key].clone() + (1 - tau) * target_value_state_dict[
                key].clone()

        t_v.load_state_dict(value_state_dict)
