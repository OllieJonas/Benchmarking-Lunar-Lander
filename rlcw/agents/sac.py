import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import util

from typing import NoReturn

from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer

DEVICE = util.get_torch_device()


def soft_copy(v, t_v, tau):
    value_state_dict = dict(v.named_parameters())
    target_value_state_dict = dict(t_v.named_parameters())

    for key in value_state_dict:
        value_state_dict[key] = tau * value_state_dict[key].clone() + (1 - tau) * target_value_state_dict[key].clone()

    t_v.load_state_dict(value_state_dict)


def hard_copy(v, t_v):
    t_v.load_state_dict(dict(v.named_parameters()))


class Value(nn.Module):

    def __init__(self, no_states, no_hidden_neurons, no_layers=1):
        super(Value, self).__init__()

        if no_layers <= 0:
            raise ValueError("can't be less than 0!")

        self.no_states = no_states
        self.no_hidden_neurons = no_hidden_neurons
        self.no_layers = no_layers
        self.initial_weight = 3e-3

        self.input_layer = nn.Linear(no_states, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)
        self.output_layer = nn.Linear(no_hidden_neurons, 1)

        self.output_layer.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

    def forward(self, state):
        inp = nn.functional.relu(self.input_layer(state))
        hidden = nn.functional.relu(self.hidden_layer(inp))
        out = self.output_layer(hidden)

        return out


class Critic(nn.Module):

    def __init__(self, no_states, no_actions, no_hidden_neurons, no_layers=1):
        super(Critic, self).__init__()

        self.no_states = no_states
        self.no_actions = no_actions
        self.no_hidden_neurons = no_hidden_neurons
        self.no_layers = no_layers
        self.initial_weight = 3e-3

        self.input_layer = nn.Linear(no_states + no_actions, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)
        self.output_layer = nn.Linear(no_hidden_neurons, 1)

        self.output_layer.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

    def forward(self, state, action):
        x = nn.functional.relu(self.input_layer(torch.cat([state, action], dim=1)))
        x = nn.functional.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x


class Actor(nn.Module):
    def __init__(self, max_action, state_shape, no_states, no_actions, no_hidden_neurons,
                 no_layers=1,
                 min_std_log=0,
                 max_std_log=1,
                 noise=1e-06):
        super(Actor, self).__init__()

        self.max_action = max_action
        self.state_shape = state_shape
        self.no_states = no_states
        self.no_actions = no_actions
        self.hidden_p = no_hidden_neurons

        self.no_layers = no_layers
        self.min_std_log = min_std_log + noise
        self.max_std_log = max_std_log
        self.noise = noise

        self.initial_weight = 3e-3

        self.input_layer = nn.Linear(*state_shape, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)

        self.mean = nn.Linear(no_hidden_neurons, no_actions)
        self.mean.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.mean.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.std = nn.Linear(no_hidden_neurons, no_actions)
        self.std.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.std.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

    def forward(self, state):
        x = nn.functional.relu(self.input_layer(state))
        x = nn.functional.relu(self.hidden_layer(x))

        mean = self.mean(x)
        std_log = torch.clamp(self.std(x), min=self.min_std_log, max=self.max_std_log)

        return mean, std_log

    def sample_normal(self, state):
        mean, std_log = self.forward(state)
        probs = torch.distributions.Normal(mean, std_log)

        actions = probs.sample()
        action = torch.tanh(actions) * torch.Tensor(self.max_action).to(DEVICE)

        log_probs = probs.log_prob(actions) - torch.log(1 - action.pow(2) + self.noise)
        log_probs = log_probs.sum(1, keepdims=True)

        return action, log_probs


class SoftActorCritic(AbstractAgent):

    def __init__(self, logger, action_space, observation_space, config):
        super().__init__(logger, action_space, config)
        self.logger.info(f'SAC Config: {config}')

        self.observation_space = observation_space
        self.action_size = action_space.n
        self.state_size = observation_space.shape[0]
        self.no_updates = 10
        self.sample_size = 10

        self.max_action = action_space.n
        self.batch_size = 10
        self.alpha = 0.9
        self.gamma = 0.9

        self.scale = 2
        self.tau = 0.005

        self.learning_rate = config["learning_rate"]
        self.no_hidden_layers = config["no_hidden_layers"]
        self.no_hidden_neurons = config["no_hidden_neurons"]

        # networks
        self.value_network = Value(no_states=self.state_size,
                                   no_hidden_neurons=self.no_hidden_neurons,
                                   no_layers=self.no_hidden_layers,
                                   ).to(DEVICE)

        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.target_value_network = Value(no_states=self.state_size,
                                          no_hidden_neurons=self.no_hidden_neurons,
                                          no_layers=self.no_hidden_layers,
                                          ).to(DEVICE)

        self.target_value_network_optimizer = optim.Adam(self.target_value_network.parameters(), lr=self.learning_rate)

        self.critic_network_1 = Critic(no_states=self.state_size,
                                       no_actions=self.action_size,
                                       no_hidden_neurons=self.no_hidden_neurons,
                                       ).to(DEVICE)

        self.critic_network_1_optimizer = optim.Adam(self.critic_network_1.parameters(), lr=self.learning_rate)

        self.critic_network_2 = Critic(no_states=self.state_size,
                                       no_actions=self.action_size,
                                       no_hidden_neurons=self.no_hidden_neurons,
                                       ).to(DEVICE)

        self.critic_network_2_optimizer = optim.Adam(self.critic_network_2.parameters(), lr=self.learning_rate)

        self.actor_network = Actor(max_action=self.max_action,
                                   state_shape=observation_space.shape,
                                   no_states=self.state_size,
                                   no_actions=self.action_size,
                                   no_hidden_neurons=self.no_hidden_neurons,
                                   ).to(DEVICE)

        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

    def name(self) -> str:
        return "SAC"

    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(DEVICE)
        actions, _ = self.actor_network.sample_normal(state)

        action = actions.cpu().detach().numpy()[0]
        self.logger.info(f"action: {action}")
        return action

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.max_capacity < self.batch_size:  # sanity check
            raise ValueError("max capacity of training_context is less than the batch size! :(")
        elif training_context.size <= self.batch_size:
            pass
        else:
            self._do_train(training_context)

    def _do_train(self, training_context: ReplayBuffer) -> NoReturn:
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

        curr_value = self.value_network.forward(curr_states)
        next_value = self.target_value_network.forward(next_states)
        next_value[dones] = 0

        new_actions, log_probs = self.actor_network.sample_normal(curr_states)
        q1_new = self.critic_network_1.forward(curr_states, new_actions)
        q2_new = self.critic_network_2.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new)

        self.value_network_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * torch.nn.functional.mse_loss(curr_value, next_value)
        value_loss.backward(retain_graph=True)
        self.value_network_optimizer.step()

        new_actions, log_probs = self.actor_network.sample_normal(curr_states)
        q1_new = self.critic_network_1.forward(curr_states, new_actions)
        q2_new = self.critic_network_2.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new)

        actor_loss = torch.mean(log_probs - critic_value)
        self.actor_network.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network.optimizer.step()

        self.critic_network_1_optimizer.zero_grad()
        self.critic_network_2_optimizer.zero_grad()
        q_hat = self.scale * rewards + self.gamma * next_value
        q1_old = self.critic_network_1.forward(curr_states, new_actions)
        q2_old = self.critic_network_2.forward(curr_states, new_actions)

        critic_1_loss = 0.5 * torch.nn.functional.mse_loss(q1_old, q_hat)
        critic_2_loss = 0.5 * torch.nn.functional.mse_loss(q2_old, q_hat)

        total_critic_loss = critic_1_loss + critic_2_loss
        total_critic_loss.backward()
        self.critic_network_1_optimizer.step()
        self.critic_network_2_optimizer.step()

        soft_copy(self.value_network, self.target_value_network, self.tau)
