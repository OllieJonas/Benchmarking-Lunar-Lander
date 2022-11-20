import random

import numpy as np
import torch
import torch.nn as nn
import util

from typing import NoReturn

from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer

DEVICE = util.get_torch_device()


class ValueNetwork(nn.Module):

    def __init__(self, psi, state_p, hidden_p, no_layers=1):
        super(ValueNetwork, self).__init__()

        self.psi = psi
        self.state_p = state_p
        self.hidden_p = hidden_p
        self.no_layers = no_layers

        self.input_layer = nn.Linear(state_p, hidden_p)

        self.hidden_layers = [nn.Linear(hidden_p, hidden_p)] * no_layers

        self.output_layer = nn.Linear(hidden_p, 1)

    def forward(self, state):
        inp = nn.functional.relu(self.input_layer(state))
        hidden = inp

        for i in range(self.no_layers):
            hidden = nn.functional.relu(self.hidden_layers[i])

        out = self.output_layer(hidden)

        return out


class SoftQNetwork(nn.Module):

    def __init__(self, theta):
        super().__init__()
        self.theta = theta


class PolicyNetwork(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau


class SoftActorCritic(AbstractAgent):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)
        self.logger.info(f'SAC Config: {config}')
        self.no_updates = 10
        self.sample_size = 10

        self.batch_size = 10
        self.alpha = 0.9
        self.gamma = 0.9

        self.no_hidden_layers = config["no_hidden_layers"]
        self.hidden_layer_size = config["hidden_layer_size"]
        # networks
        value_network = ValueNetwork(psi=0.01, no_layers=self.no_hidden_layers).to(DEVICE)

    def name(self) -> str:
        return "SAC"

    def get_action(self, state):
        return random.choice(range(self.action_space.n))

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.max_capacity < self.batch_size:  # sanity check
            raise ValueError("max capacity of training_context is less than the batch size! :(")
        elif training_context.size <= self.batch_size:
            pass
        else:
            self._train(training_context)

    def _train(self, training_context: ReplayBuffer) -> NoReturn:
        random_sample = training_context.random_sample(self.sample_size)

        random_sample = np.fromiter(((s["curr_state"], s["next_state"], s["reward"],
                                      s["action"], s["terminated"]) for s in random_sample), dtype=random_sample.dtype)

        # REMOVE ABOVE WHEN CHANGING FROM DICT TO LIST

        # TODO: This is literal hot garbage. Please fix. Thanks! :)
        curr_states, next_states, rewards, actions, dones = [np.asarray(x) for x in zip(*random_sample)]

        curr_states = torch.from_numpy(curr_states).to(DEVICE)
        next_states = torch.from_numpy(next_states).to(DEVICE)
        rewards = torch.from_numpy(rewards).to(DEVICE)
        actions = torch.from_numpy(actions).to(DEVICE)
        dones = torch.from_numpy(dones).to(DEVICE)

    def _compute_targets(self, reward, next_state, terminated):
        return reward + ((self.gamma * (1 - terminated)) * ())
