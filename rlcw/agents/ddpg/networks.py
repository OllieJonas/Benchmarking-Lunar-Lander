import numpy as np
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, no_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.no_actions = no_actions

        # input_layer, input_batch_norm = init_linear_layer(*input_dims, fc1_dims)
        # hidden_layer, hidden_batch_norm = init_linear_layer(fc1_dims, fc2_dims)
        # output_layer, _ = init_linear_layer(fc2_dims, no_actions, weight_bias_range=0.003)

        # self.layers = nn.Sequential(
        #     input_layer,
        #     input_batch_norm,
        #     nn.ReLU(),
        #     hidden_layer,
        #     hidden_batch_norm,
        #     nn.ReLU(),
        #     output_layer,
        #     nn.Tanh(),
        # )

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        fc1_range = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -fc1_range, fc1_range)
        nn.init.uniform_(self.fc1.bias.data, -fc1_range, fc1_range)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        fc2_range = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -fc2_range, fc2_range)
        nn.init.uniform_(self.fc2.bias.data, -fc2_range, fc2_range)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        f3 = 0.003
        self.mu = nn.Linear(self.fc2_dims, self.no_actions)
        nn.init.uniform_(self.mu.weight.data, -f3, f3)
        nn.init.uniform_(self.mu.bias.data, -f3, f3)

    def forward(self, state):

        x = self.fc1(state)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = torch.tanh(self.mu(x))

        # return self.layers.forward(x)
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims, no_actions):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.no_actions = no_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        fc1_range = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        nn.init.uniform_(self.fc1.weight.data, -fc1_range, fc1_range)
        nn.init.uniform_(self.fc1.bias.data, -fc1_range, fc1_range)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        fc2_range = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        nn.init.uniform_(self.fc2.weight.data, -fc2_range, fc2_range)
        nn.init.uniform_(self.fc2.bias.data, -fc2_range, fc2_range)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.no_actions, self.fc2_dims)
        f3 = 0.003
        self.q = nn.Linear(self.fc2_dims, 1)
        nn.init.uniform_(self.q.weight.data, -f3, f3)
        nn.init.uniform_(self.q.bias.data, -f3, f3)

        # input_layer, input_batch_norm = init_linear_layer(*input_dims, fc1_dims)
        # hidden_layer, hidden_batch_norm = init_linear_layer(fc1_dims, fc2_dims)
        # action_value = nn.Linear(no_actions, fc2_dims)
        # q_layer, _ = init_linear_layer(fc2_dims, 1, weight_bias_range=0.003)

        # self.state_value_layers = nn.Sequential(
        #     input_layer,
        #     input_batch_norm,
        #     nn.ReLU(),
        #     hidden_layer,
        #     hidden_batch_norm,
        # )
        #
        # self.action_value_layers = nn.Sequential(
        #     action_value,
        #     nn.ReLU(),
        # )
        #
        # self.state_action_value_layers = nn.Sequential(
        #     nn.ReLU(),
        #     q_layer
        # )

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = nn.functional.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = nn.functional.relu(self.action_value(action))
        state_action_value = nn.functional.relu(torch.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        # state_value = self.state_value_layers.forward(state)
        # action_value = self.action_value_layers.forward(action)

        # return self.state_action_value_layers.forward(torch.add(state_value, action_value))

        return state_action_value


def init_linear_layer(input_dims, output_dims, weight_bias_range=None):
    layer = nn.Linear(input_dims, output_dims)
    if weight_bias_range is None:
        weight_bias_range = 1. / np.sqrt(layer.weight.data.size()[0])

    nn.init.uniform_(layer.weight.data, -weight_bias_range, weight_bias_range)
    nn.init.uniform_(layer.bias.data, -weight_bias_range, weight_bias_range)

    return layer, nn.LayerNorm(output_dims)
