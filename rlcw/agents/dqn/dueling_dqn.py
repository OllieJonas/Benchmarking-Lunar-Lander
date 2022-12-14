import copy
import torch
import numpy as np
from typing import NoReturn

import agents.common.utils as agent_utils
from agents.dqn.dqn import DQN
from agents.dqn.networks import DuelingDeepQNetwork
from agents.dqn.policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer


class DuelingDQN(DQN):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        # config vars

        self.no_actions = None
        self.q, self.q_optim = None, None
        self.target_q = None

    def assign_env_dependent_variables(self, action_space, state_space):
        self.no_actions = action_space.n

        self.q, self.q_optim = agent_utils.with_optim(
            DuelingDeepQNetwork(*state_space.shape, self.no_actions, hidden_dims=self.hidden_size),
            self.learning_rate, device=self.device)

        self.target_q = DuelingDeepQNetwork(*state_space.shape, self.no_actions, hidden_dims=self.hidden_size).to(
            self.device)

        self.policy = EpsilonGreedyPolicy(self.epsilon, self.no_actions, self.device)

    def name(self) -> str:
        return "DuelingDQN"

    def save(self):
        self.save_checkpoint(self.q, "Q")
        self.save_checkpoint(self.target_q, "TargetQ")

    def load(self, path):
        self.load_checkpoint(self.q, path, "Q")
        self.load_checkpoint(self.q, path, "TargetQ")

    def get_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        advantage, _ = self.q.forward(state)
        return self.policy.get_action(advantage)

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.cnt >= self.batch_size:
            self._do_train(training_context)

    def _do_train(self, training_context):
        states, actions, rewards, next_states, dones = \
            training_context.random_sample(self.batch_size)

        self.q_optim.zero_grad()

        if self._update_cnt % self.update_count == 0:
            self._sync_target_network()

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.int32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        indices = np.arange(self.batch_size)

        q_curr_advantage, q_curr_value = self.q.forward(states)
        q_next_advantage, q_next_value = self.target_q.forward(next_states)

        q_curr = torch.add(q_curr_value, self._calculate_advantage(q_curr_advantage))[indices, actions]
        q_next = torch.add(q_next_value, self._calculate_advantage(q_next_advantage)).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next

        loss = self.criterion(q_target, q_curr).to(self.device)
        loss.backward()
        self.q_optim.step()
        self._update_cnt += 1

        self.decay_epsilon()

    @staticmethod
    def _calculate_advantage(q_advantage):
        return q_advantage - q_advantage.mean(dim=1, keepdim=True)

    def _sync_target_network(self):
        self.logger.info(f"Epsilon: {self.epsilon}")
        self.q = copy.deepcopy(self.target_q)
