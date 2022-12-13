import copy
from typing import NoReturn

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agents.abstract_agent import CheckpointAgent
from agents.dqn.policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, no_actions, hidden_1_dims=256, hidden_2_dims=256):
        super(DeepQNetwork, self).__init__()
        self.input_size = input_size
        self.no_actions = no_actions

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_1_dims),
            nn.ReLU(),
            nn.Linear(hidden_1_dims, hidden_2_dims),
            nn.ReLU(),
            nn.Linear(hidden_2_dims, no_actions)
        )

    def forward(self, state):
        return self.layers.forward(state)


class DQN(CheckpointAgent):

    def __init__(self, logger, action_space, state_space, config):
        super().__init__(logger, action_space, config)

        # config vars

        self.learning_rate = config["learning_rate"]
        self.batch_size = config["batch_size"]
        self.hidden_size = config["hidden_layer_size"]
        self.update_count = config["update_count"]

        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon_decay"]
        self.epsilon_min = config["epsilon_min"]
        self.gamma = config["gamma"]

        self.logger.info(f"DQN Config: {config}")

        self.no_actions = action_space.n

        self.q_network = DeepQNetwork(*state_space.shape, self.no_actions,
                                      hidden_1_dims=self.hidden_size, hidden_2_dims=self.hidden_size)\
            .to(self.device)

        self.target_q_network = DeepQNetwork(*state_space.shape, self.no_actions,
                                             hidden_1_dims=self.hidden_size, hidden_2_dims=self.hidden_size)\
            .to(self.device)

        self._sync_target_network()

        self.value_network_optimiser = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.criterion = nn.HuberLoss()

        self._batch_cnt = 0
        self._update_cnt = 0

        self.policy = EpsilonGreedyPolicy(self.epsilon, self.no_actions, self.device)

    def _sync_target_network(self):
        self.logger.info(f"Epsilon: {self.epsilon}")
        self.target_q_network = copy.deepcopy(self.q_network)

    def save(self):
        self.save_checkpoint(self.q_network, "ValueNetwork")
        self.save_checkpoint(self.target_q_network, "TargetValueNetwork")

    def load(self, path):
        self.load_checkpoint(self.q_network, path, "ValueNetwork")
        self.load_checkpoint(self.target_q_network, path, "TargetValueNetwork")

    def name(self) -> str:
        return "DQN"

    def get_action(self, state):
        if np.random.random() > self.epsilon:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            actions = self.q_network.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(range(self.action_space.n))

        return action

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.cnt >= self.batch_size:
            self._do_train(training_context)

    def _do_train(self, training_context):
        states, actions, rewards, next_states, dones = \
            training_context.random_sample(self.batch_size)

        self.value_network_optimiser.zero_grad()

        if self._update_cnt % self.update_count == 0:
            self._sync_target_network()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        indices = np.arange(self.batch_size)

        q_curr = self.q_network.forward(states)[indices, actions]
        q_next = self.target_q_network.forward(next_states).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.criterion(q_target, q_curr).to(self.device)
        loss.backward()
        self.value_network_optimiser.step()
        self._update_cnt += 1

        self._decay_epsilon()

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
