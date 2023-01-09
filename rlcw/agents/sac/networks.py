import torch
import torch.nn as nn


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

        self.output_layer.weight.data.uniform_(
            -self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(
            -self.initial_weight, +self.initial_weight)

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

        self.output_layer.weight.data.uniform_(
            -self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(
            -self.initial_weight, +self.initial_weight)

        self.float()

    def forward(self, state, action):
        x = nn.functional.relu(self.input_layer(
            torch.cat([state, action], dim=1)))
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

        self.device = device
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
        self.mean.weight.data.uniform_(-self.initial_weight, +
                                       self.initial_weight)
        self.mean.bias.data.uniform_(-self.initial_weight, +
                                     self.initial_weight)

        self.std = nn.Linear(no_hidden_neurons, no_actions)
        self.std.weight.data.uniform_(-self.initial_weight, +
                                      self.initial_weight)
        self.std.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.float()

    def forward(self, state):
        x = nn.functional.relu(self.input_layer(state))
        x = nn.functional.relu(self.hidden_layer(x))
        mean = self.mean(x)
        std_log = torch.clamp(
            self.std(x), min=self.min_std_log, max=self.max_std_log)

        return mean, std_log

    def sample_normal(self, state):

        mean, std_log = self.forward(state)
        probs = torch.distributions.Normal(mean, std_log)

        normal_sample = probs.sample()

        action = torch.tanh(normal_sample) * \
            torch.Tensor(self.max_action).to(self.device)
        log_probs = (probs.log_prob(normal_sample) -
                     torch.log(1 - action.pow(2) + self.noise)).sum(1)
        return action, log_probs
