"""
author: Yatin & Ollie
"""
from typing import NoReturn

import torch
import torch.nn as nn
import torch.optim as optim

import util
from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer

class Value(nn.Module):

    def __init__(self, no_states, no_hidden_neurons, no_layers=1, initial_weight=3e-3):
        super(Value, self).__init__()

        if no_layers <= 0:
            raise ValueError("can't be less than 0!")

        self.no_states = no_states
        self.no_hidden_neurons = no_hidden_neurons
        self.no_layers = no_layers
        self.initial_weight = initial_weight

        self.input_layer = nn.Linear(no_states, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)
        self.output_layer = nn.Linear(no_hidden_neurons, 1)

        self.output_layer.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.float()

    def forward(self, state):
        inp = nn.functional.relu(self.input_layer(state))
        hidden = nn.functional.relu(self.hidden_layer(inp))
        out = self.output_layer(hidden)

        return out


class Critic(nn.Module):

    def __init__(self, no_states, no_actions, no_hidden_neurons, no_layers=1, initial_weight=3e-3):
        super(Critic, self).__init__()

        self.no_states = no_states
        self.no_actions = no_actions
        self.no_hidden_neurons = no_hidden_neurons
        self.no_layers = no_layers
        self.initial_weight = initial_weight

        self.input_layer = nn.Linear(no_states + no_actions, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)
        self.output_layer = nn.Linear(no_hidden_neurons, 1)

        self.output_layer.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.float()

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
                 noise=1e-06,
                 initial_weight=3e-3):
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

        self.initial_weight = initial_weight

        self.input_layer = nn.Linear(*state_shape, no_hidden_neurons)
        self.hidden_layer = nn.Linear(no_hidden_neurons, no_hidden_neurons)

        self.mean = nn.Linear(no_hidden_neurons, no_actions)
        self.mean.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.mean.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.std = nn.Linear(no_hidden_neurons, no_actions)
        self.std.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.std.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.float()

    def forward(self, state):
        x = nn.functional.relu(self.input_layer(state))
        x = nn.functional.relu(self.hidden_layer(x))

        mean = self.mean(x)
        std_log = torch.clamp(self.std(x), min=self.min_std_log, max=self.max_std_log)

        return mean, std_log

    def sample_normal(self, state, reparameterise=True):
        mean, std_log = self.forward(state)
        probs = torch.distributions.Normal(mean, std_log)

        normal_sample = probs.sample()

        action = torch.tanh(normal_sample) * torch.Tensor(self.max_action).to(self.device)

        log_probs = (probs.log_prob(normal_sample) - torch.log(1 - action.pow(2) + self.noise)).sum(1)
        return action, log_probs


class SoftActorCritic(CheckpointAgent):

    def __init__(self, logger, action_space, observation_space, config):
        super().__init__(logger, action_space, config)
        self.logger.info(f'SAC Config: {config}')

        self.observation_space = observation_space
        self.action_size = action_space.shape[0]
        self.state_size = observation_space.shape[0]
        self.max_action = action_space.high

        self._batch_cnt = 0

        # hyperparams
        # batches
        self.sample_size = config["sample_size"]
        self.batch_size = config["batch_size"]

        # algo
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]

        self.scale = config["scale"]
        self.tau = config["tau"]

        # nn
        self.nn_initial_weights = config["nn_initial_weights"]
        self.actor_noise = config["actor_noise"]
        self.learning_rate = config["learning_rate"]
        self.no_hidden_neurons = config["no_hidden_neurons"]

        # networks
        self.value_network = Value(no_states=self.state_size,
                                   no_hidden_neurons=self.no_hidden_neurons,
                                   initial_weight=self.nn_initial_weights,
                                   ).to(self.device)

        self.value_network_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)

        self.target_value_network = Value(no_states=self.state_size,
                                          no_hidden_neurons=self.no_hidden_neurons,
                                          initial_weight=self.nn_initial_weights,
                                          ).to(self.device)

        self.target_value_network_optimizer = optim.Adam(self.target_value_network.parameters(), lr=self.learning_rate)

        self.critic_network_1 = Critic(no_states=self.state_size,
                                       no_actions=self.action_size,
                                       no_hidden_neurons=self.no_hidden_neurons,
                                       initial_weight=self.nn_initial_weights,
                                       ).to(self.device)

        self.critic_network_1_optimizer = optim.Adam(self.critic_network_1.parameters(), lr=self.learning_rate)

        self.critic_network_2 = Critic(no_states=self.state_size,
                                       no_actions=self.action_size,
                                       no_hidden_neurons=self.no_hidden_neurons,
                                       initial_weight=self.nn_initial_weights,
                                       ).to(self.device)

        self.critic_network_2_optimizer = optim.Adam(self.critic_network_2.parameters(), lr=self.learning_rate)

        self.actor_network = Actor(max_action=self.max_action,
                                   state_shape=observation_space.shape,
                                   no_states=self.state_size,
                                   no_actions=self.action_size,
                                   no_hidden_neurons=self.no_hidden_neurons,
                                   initial_weight=self.nn_initial_weights,
                                   ).to(self.device)

        self.actor_network_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.learning_rate)

    def save(self):
        self.save_checkpoint(self.value_network, "Value")
        self.save_checkpoint(self.target_value_network, "TargetValue")
        self.save_checkpoint(self.critic_network_1, "Critic1")
        self.save_checkpoint(self.critic_network_2, "Critic2")
        self.save_checkpoint(self.actor_network, "Actor")

    def load(self, path):
        self.load_checkpoint(self.value_network, path, "Value")
        self.load_checkpoint(self.target_value_network, path, "TargetValue")
        self.load_checkpoint(self.critic_network_1, path, "Critic1")
        self.load_checkpoint(self.critic_network_2, path, "Critic2")
        self.load_checkpoint(self.actor_network, path, "Actor")

    def name(self) -> str:
        return "SAC"

    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        actions, _ = self.actor_network.sample_normal(state)

        action = actions.cpu().detach().numpy()[0]
        return action

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        if training_context.max_capacity < self.batch_size:  # sanity check
            raise ValueError("max capacity of training_context is less than the batch size! :(")

        if self._batch_cnt <= self.batch_size:
            self._batch_cnt += 1
            return
        else:
            self._batch_cnt = 0
            self._do_train(training_context)

    @staticmethod
    def _soft_copy(v, t_v, tau):
        value_state_dict = dict(v.named_parameters())
        target_value_state_dict = dict(t_v.named_parameters())

        for key in value_state_dict:
            value_state_dict[key] = tau * value_state_dict[key].clone() + (1 - tau) * target_value_state_dict[
                key].clone()

        t_v.load_state_dict(value_state_dict)

    @staticmethod
    def _hard_copy(v, t_v):
        t_v.load_state_dict(dict(v.named_parameters()))

    def _do_train(self, training_context: ReplayBuffer) -> NoReturn:
        curr_states, next_states, rewards, actions, dones \
            = training_context.random_sample_transformed(self.sample_size, self.device)

        curr_value = self.value_network.forward(curr_states).view(-1)
        next_value = self.target_value_network.forward(next_states).view(-1)

        new_actions, log_probs = self.actor_network.sample_normal(curr_states, reparameterise=False)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_network_1.forward(curr_states, new_actions)
        q2_new = self.critic_network_2.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new).view(-1)

        self.value_network_optimizer.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * torch.nn.functional.mse_loss(curr_value, next_value)
        value_loss.backward(retain_graph=True)
        self.value_network_optimizer.step()

        new_actions, log_probs = self.actor_network.sample_normal(curr_states, reparameterise=True)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_network_1.forward(curr_states, new_actions)
        q2_new = self.critic_network_2.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new).view(-1)

        actor_loss = torch.mean(log_probs - critic_value)
        self.actor_network_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_network_optimizer.step()

        self.critic_network_1_optimizer.zero_grad()
        self.critic_network_2_optimizer.zero_grad()
        q_hat = self.scale * rewards + self.gamma * next_value
        q1_old = self.critic_network_1.forward(curr_states, new_actions).view(-1)
        q2_old = self.critic_network_2.forward(curr_states, new_actions).view(-1)

        critic_1_loss = 0.5 * torch.nn.functional.mse_loss(q1_old, q_hat)
        critic_2_loss = 0.5 * torch.nn.functional.mse_loss(q2_old, q_hat)

        total_critic_loss = critic_1_loss + critic_2_loss
        total_critic_loss.backward()
        self.critic_network_1_optimizer.step()
        self.critic_network_2_optimizer.step()

        self._soft_copy(self.value_network, self.target_value_network, self.tau)
