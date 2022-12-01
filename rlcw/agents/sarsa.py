"""
author: Helen
"""
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim

import util
from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer

DEVICE = util.get_torch_device()


# neural network as q-table for lunar landing to too big
class ActionValueNetwork(nn.Module):

    def __init__(self, state_space, action_space, no_layers=1, initial_weight=3e-3):
        super().__init__()

        if no_layers <= 0:
            raise ValueError("can't be less than 0!")

        self.no_states = state_space.shape[0]
        self.no_hidden_neurons = 256
        self.no_layers = no_layers
        self.initial_weight = initial_weight

        self.input_layer = nn.Linear(self.no_states, self.no_hidden_neurons)
        self.hidden_layer = nn.Linear(self.no_hidden_neurons, self.no_hidden_neurons)
        self.output_layer = nn.Linear(self.no_hidden_neurons, action_space.n)

        self.output_layer.weight.data.uniform_(-self.initial_weight, +self.initial_weight)
        self.output_layer.bias.data.uniform_(-self.initial_weight, +self.initial_weight)

        self.float()

    def forward(self, state):
        state = torch.Tensor(state)
        inp = nn.functional.relu(self.input_layer(state))
        hidden = nn.functional.relu(self.hidden_layer(inp))
        out = self.output_layer(hidden)

        return out


# Technically Deep/expected SARSA but t'is late so just SARSA
class SarsaAgent(CheckpointAgent):
    def name(self):
        return "sarsa"

    def __init__(self, logger, action_space, state_space, config):
        super().__init__(logger, action_space, config)

        self.network = ActionValueNetwork(state_space, action_space).to(DEVICE)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001,
                                    betas=(0.001, 0.00199),
                                    eps=1e-8)
        self.criterion = nn.MSELoss()

        self.num_actions = action_space.n

        self.num_replay = 8

        # config vars
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.sample_size = config["sample_size"]
        self.batch_size = config["batch_size"]

        self.loss = 0

    def save(self):
        pass

    def load(self, path):
        pass

    def get_action(self, state):

        action_values = self.network.forward(state)
        probs_batch = self.softmax(action_values, self.tau).detach().numpy()

        action = np.random.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def train(self, training_context: ReplayBuffer):

        if training_context.cnt < self.replay_buffer.batch_size:
            return

        current_q = deepcopy(self.network)
        for _ in range(self.num_replay):
            experiences = training_context.random_sample(self.sample_size)

            self.loss += self.optimize_network(experiences, self.gamma, self.optimizer, self.network, current_q,
                                               self.tau,
                                               self.criterion)

    def softmax(self, action_values, tau=1.0):
        """
        Args:
            action_values (Tensor array): A 2D array of shape (batch_size, num_actions).
                        The action-values computed by an action-value network.
            tau (float): The temperature parameter scalar.
        Returns:
            A 2D Tensor array of shape (batch_size, num_actions). Where each column is a probability distribution
            over the actions representing the get_action.
        """
        preferences = action_values / tau
        max_preference = torch.max(preferences)

        reshaped_max_preference = max_preference.view((-1, 1))
        exp_preferences = torch.exp(preferences - reshaped_max_preference)
        sum_of_exp_preferences = torch.sum(exp_preferences, dim=1)

        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.view((-1, 1))

        action_probs = exp_preferences / reshaped_sum_of_exp_preferences
        action_probs = action_probs.squeeze()

        return action_probs

    def get_td(self, states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
        """
        Args:
            states (Numpy array): The batch of states with the shape (batch_size, state_dim).
            next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
            actions (Numpy array): The batch of actions with the shape (batch_size,).
            rewards (Numpy array): The batch of rewards with the shape (batch_size,).
            discount (float): The discount factor (gamma).
            terminals (Numpy array): The batch of terminals with the shape (batch_size,).
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                            and particularly, the action-values at the next-states.
        Returns:
            target_vec (Tensor array): The TD Target for actions taken, of shape (batch_size,)
            estimate_vec (Tensor array): The TD estimate for actions taken, of shape (batch_size,)
        """

        q_next_mat = current_q.forward(next_states).detach()

        probs_mat = self.softmax(q_next_mat, tau)
        v_next_vec = torch.zeros((q_next_mat.shape[0]), dtype=torch.float64).detach()

        if terminals.any() == 1:
            terminals = [1]
        else:
            terminals = [0]
        v_next_vec = torch.sum(probs_mat * q_next_mat, dim=1) * (1 - torch.tensor(terminals))

        target_vec = torch.tensor(rewards) + (discount * v_next_vec)

        q_mat = network.forward(states)

        batch_indices = torch.arange(q_mat.shape[0])

        estimate_vec = q_mat[batch_indices, np.expand_dims(np.max(actions, axis=1), axis=1)]

        return target_vec, estimate_vec

    def optimize_network(self, experiences, discount, optimizer, network, current_q, tau, criterion):
        """
        Args:
            experiences (Numpy array): The batch of experiences including the states, actions,
                                    rewards, terminals, and next_states.
            discount (float): The discount factor.
            network (ActionValueNetwork): The latest state of the network that is getting replay updates.
            current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                            and particularly, the action-values at the next-states.
        Return:
            Loss (float): The loss value for the current batch.
        """
        states = experiences[0]
        actions = experiences[1]
        rewards = experiences[2]
        next_states = experiences[3]

        terminals = experiences[4]
        batch_size = states.shape[0]

        td_target, td_estimate = self.get_td(states, next_states, actions, rewards, discount, terminals, \
                                             network, current_q, tau)

        loss = criterion(td_estimate.double().to(DEVICE), td_target.to(DEVICE))
        loss.backward()

        return (loss / batch_size).detach().numpy()