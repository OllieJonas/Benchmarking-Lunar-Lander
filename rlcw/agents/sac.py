"""
author: Yatin & Ollie
"""
from typing import NoReturn

import agents.common.utils as agent_utils
from agents.abstract_agent import CheckpointAgent
import torch
import torch.nn as nn

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
    def __init__(self, device, max_action, state_shape, no_states, no_actions, no_hidden_neurons,
                 no_layers=1,
                 min_std_log=0,
                 max_std_log=1,
                 noise=1e-06,
                 initial_weight=3e-3):
        super(Actor, self).__init__()

        self.device=device
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

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.logger.info(f'SAC Config: {config}')
        self._batch_cnt = 0

        self.requires_continuous_action_space = True
                
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

        self.state_size = 0
        self.max_action = 0
        self.action_size = 0

        self.value, self.value_optim = None, None
        self.target_value, self.target_value_optim = None, None
        self.critic_one, self.critic_one_optim = None, None
        self.critic_two, self.critic_two_optim = None, None
        self.actor, self.actor_optim = None, None

    def assign_env_dependent_variables(self, action_space, state_space):
        self.action_size = action_space.shape[0]
        self.state_size = state_space.shape[0]
        self.max_action = action_space.high

        # networks
        self.value, self.value_optim = agent_utils.with_optim(
            Value(no_states=self.state_size,
                  no_hidden_neurons=self.no_hidden_neurons,
                  initial_weight=self.nn_initial_weights,
                  ), self.learning_rate, device=self.device)

        self.target_value, self.target_value_optim = agent_utils.with_optim(
            Value(no_states=self.state_size,
                  no_hidden_neurons=self.no_hidden_neurons,
                  initial_weight=self.nn_initial_weights,
                  ), self.learning_rate, device=self.device)

        self.critic_one, self.critic_one_optim = agent_utils.with_optim(
            Critic(no_states=self.state_size,
                   no_actions=self.action_size,
                   no_hidden_neurons=self.no_hidden_neurons,
                   initial_weight=self.nn_initial_weights,
                   ), self.learning_rate, device=self.device)

        self.critic_two, self.critic_two_optim = agent_utils.with_optim(
            Critic(no_states=self.state_size,
                   no_actions=self.action_size,
                   no_hidden_neurons=self.no_hidden_neurons,
                   initial_weight=self.nn_initial_weights,
                   ), self.learning_rate, device=self.device)

        self.actor, self.actor_optim = agent_utils.with_optim(
            Actor(device=self.device,
                  max_action=self.max_action,
                  state_shape=state_space.shape,
                  no_states=self.state_size,
                  no_actions=self.action_size,
                  no_hidden_neurons=self.no_hidden_neurons,
                  initial_weight=self.nn_initial_weights,
                  ), self.learning_rate, device=self.device)

    def save(self):
        self.save_checkpoint(self.value, "Value")
        self.save_checkpoint(self.target_value, "TargetValue")
        self.save_checkpoint(self.critic_one, "Critic1")
        self.save_checkpoint(self.critic_two, "Critic2")
        self.save_checkpoint(self.actor, "Actor")

    def load(self, path):
        self.load_checkpoint(self.value, path, "Value")
        self.load_checkpoint(self.target_value, path, "TargetValue")
        self.load_checkpoint(self.critic_one, path, "Critic1")
        self.load_checkpoint(self.critic_two, path, "Critic2")
        self.load_checkpoint(self.actor, path, "Actor")

    def name(self) -> str:
        return "SAC"

    def get_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        actions, _ = self.actor.sample_normal(state, reparameterise=False)

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

    def _do_train(self, training_context: ReplayBuffer) -> NoReturn:
        curr_states, next_states, rewards, actions, dones \
            = training_context.random_sample_as_tensors(self.sample_size, self.device)

        curr_value = self.value.forward(curr_states).view(-1)
        next_value = self.target_value.forward(next_states).view(-1)

        new_actions, log_probs = self.actor.sample_normal(curr_states, reparameterise=False)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_one.forward(curr_states, new_actions)
        q2_new = self.critic_two.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new).view(-1)

        self.value_optim.zero_grad()
        value_target = critic_value - log_probs
        value_loss = 0.5 * torch.nn.functional.mse_loss(curr_value, next_value)
        value_loss.backward(retain_graph=True)
        self.value_optim.step()

        new_actions, log_probs = self.actor.sample_normal(curr_states, reparameterise=True)
        log_probs = log_probs.view(-1)
        q1_new = self.critic_one.forward(curr_states, new_actions)
        q2_new = self.critic_two.forward(curr_states, new_actions)
        critic_value = torch.min(q1_new, q2_new).view(-1)

        actor_loss = torch.mean(log_probs - critic_value)
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_one_optim.zero_grad()
        self.critic_two_optim.zero_grad()
        q_hat = self.scale * rewards + self.gamma * next_value
        q1_old = self.critic_one.forward(curr_states, new_actions).view(-1)
        q2_old = self.critic_two.forward(curr_states, new_actions).view(-1)

        critic_one_loss = 0.5 * torch.nn.functional.mse_loss(q1_old, q_hat)
        critic_two_loss = 0.5 * torch.nn.functional.mse_loss(q2_old, q_hat)

        total_critic_loss = critic_one_loss + critic_two_loss
        total_critic_loss.backward()
        self.critic_one_optim.step()
        self.critic_two_optim.step()

        agent_utils.soft_copy(self.value, self.target_value, self.tau)
