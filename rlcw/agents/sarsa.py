"""
author: Helen
"""
import random

import numpy as np

from typing import NoReturn, List

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
import util
from agents.abstract_agent import CheckpointAgent
from replay_buffer import ReplayBuffer

'''
 Neural Network to store state action values
'''
class StateActionNetwork(nn.Module):
    def __init__(self, state_dim , action_dim):
        super(StateActionNetwork, self).__init__()
        self.num_hidden_units =  256

        self.hidden_layer = nn.Linear(state_dim, self.num_hidden_units)
        self.output_layer = nn.Linear(self.num_hidden_units, action_dim)    

    def forward(self, state):
        q_vals = F.relu(self.hidden_layer(state))
        q_vals = self.output_layer(q_vals)

        return q_vals


'''
    Sarsa agent to solve Lunar lander
'''
class SarsaAgent(CheckpointAgent):

    def name(self):
        return "sarsa"
    
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.batch_size = config["batch_size"]
        self.epsilon = config["epsilon"]
        self.gamma = config["gamma"]
        self.learning_rate = config["learning_rate"]
        self.MSELoss_function = nn.MSELoss()


    def assign_env_dependent_variables(self, action_space, state_space):
        state_space = state_space.shape[0]
        action_space = action_space.n

        self.network = StateActionNetwork(state_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= 0.995 # decays epsilon 
    
        if self.epsilon <= 0.1:
            self.epsilon = 0.1

    def get_action(self, state):

        if np.random.uniform(0, 1) < self.epsilon:
                return self.action_space.sample() 
        else:
                if len(state) == 2:
                    state = state[0]
                
                state = torch.from_numpy(state).float()
                network_output_to_numpy = self.network(state).data.numpy()
                return np.argmax(network_output_to_numpy) 

    def update_Sarsa_Network(self, state, next_state, action, next_action, reward, terminals):

        q_action = torch.gather(self.network(state), dim=1, index=action.long())

        q_next_action = torch.gather(self.network(next_state), dim=1, index=next_action.long())

        qsa_next_target = reward + (self.gamma * q_next_action) * (1 - terminals)
        q_network_loss = self.MSELoss_function(q_action, qsa_next_target.detach())
        self.optimizer.zero_grad()
        q_network_loss.backward()
        self.optimizer.step()
            
    def save(self):
        self.save_checkpoint(self.network, "StateActionNetwork")

    def load(self, path):
        self.load_checkpoint(self.network, path, "StateActionNetwork")
    
    def train(self, training_context: ReplayBuffer):

        states, next_states, actions, next_actions, rewards, terminals = training_context.random_sample_sarsa(self.batch_size)
        states = torch.Tensor(states)
        next_states = torch.Tensor(next_states)
        actions = torch.Tensor(actions)
        next_actions = torch.Tensor(next_actions)
        rewards = torch.Tensor(rewards)
        terminals = torch.Tensor(terminals)
        self.update_Sarsa_Network(states, next_states, actions, next_actions, rewards, terminals)
