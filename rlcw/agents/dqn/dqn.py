import copy
from typing import NoReturn

import numpy as np
import torch
import torch.nn as nn

import agents.common.utils as agent_utils
from agents.dqn.networks import DeepQNetwork
from agents.abstract_agent import CheckpointAgent
from agents.dqn.policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer


class DQN(CheckpointAgent):

    def __init__(self, logger, config):
        super().__init__(logger, config)

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

        self.criterion = nn.HuberLoss()

        self._batch_cnt = 0
        self._update_cnt = 0

        self.policy = None
        self._q, self._q_optim = None, None
        self._target_q = None
        self.no_actions = None

    def assign_env_dependent_variables(self, action_space, state_space):
        self.no_actions = action_space.n

        self._q, self._q_optim = agent_utils.with_optim(
            DeepQNetwork(*state_space.shape, self.no_actions,
                         hidden_1_dims=self.hidden_size,
                         hidden_2_dims=self.hidden_size),
            self.learning_rate, device=self.device)

        self._target_q = DeepQNetwork(*state_space.shape, self.no_actions,
                                      hidden_1_dims=self.hidden_size, hidden_2_dims=self.hidden_size) \
            .to(self.device)

        self._sync_target_network()

        self.policy = EpsilonGreedyPolicy(self.epsilon, self.no_actions, self.device)

    def _sync_target_network(self):
        self.logger.info(f"Epsilon: {self.epsilon}")
        self._target_q = copy.deepcopy(self._q)

    def save(self):
        self.save_checkpoint(self._q, "ValueNetwork")
        self.save_checkpoint(self._target_q, "TargetValueNetwork")

    def load(self, path):
        self.load_checkpoint(self._q, path, "ValueNetwork")
        self.load_checkpoint(self._target_q, path, "TargetValueNetwork")

    def name(self) -> str:
        return "DQN"

    def get_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        actions = self._q.forward(state)
        return self.policy.get_action(actions)

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.cnt >= self.batch_size:
            self._do_train(training_context)

    def _do_train(self, training_context):
        states, actions, rewards, next_states, dones = \
            training_context.random_sample(self.batch_size)

        self._q_optim.zero_grad()

        if self._update_cnt % self.update_count == 0:
            self._sync_target_network()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        indices = np.arange(self.batch_size)

        q_curr = self._q.forward(states)[indices, actions]
        q_next = self._target_q.forward(next_states).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.criterion(q_target, q_curr).to(self.device)
        loss.backward()
        self._q_optim.step()
        self._update_cnt += 1

        self.decay_epsilon()

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
