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
# neural network as q-table for lunar landing to too big
class ActionValue(nn.Module):
    def __init__(self, state_dim , action_dim):
        super(ActionValue, self).__init__()
        #import pdb; pdb.set_trace();
        self.num_hidden_units =  256

        self.hidden_layer = nn.Linear(state_dim, self.num_hidden_units)
        self.output_layer = nn.Linear(self.num_hidden_units, action_dim)    
        '''    
        self.x_layer = nn.Linear(state_dim.shape[0], 150)
        self.h_layer = nn.Linear(150, 120)
        self.y_layer = nn.Linear(120, action_dim.n)
        print(self.x_layer)
        '''

    def forward(self, state):
        '''
        xh = F.relu(self.x_layer(state))
        hh = F.relu(self.h_layer(xh))
 
        state_action_values = self.y_layer(hh)
        return state_action_values       
        '''
        q_vals = F.relu(self.hidden_layer(state))
        q_vals = self.output_layer(q_vals)

        return q_vals


# Technically Deep/expected SARSA but t'is late so just SARSA
class SarsaAgent(CheckpointAgent):

    def name(self):
        return "sarsa"
    
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.sample_size = config["sample_size"]
        self.batch_size = config["batch_size"]
        self.epsilon = 0.99
        self.discount_factor = 0.99
        self.MSELoss_function = nn.MSELoss()


    def assign_env_dependent_variables(self, action_space, state_space):
        state_space = state_space.shape[0]
        action_space = action_space.n

        self.network = ActionValue(state_space, action_space).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def decay_epsilon(self):
        if self.epsilon > 0.2:
            self.epsilon *= 0.995 #change this so eplison does not decay as fast
    
        if self.epsilon <= 0.2:
            self.epsilon = 0.2

    def get_action(self, state):

        if np.random.uniform(0, 1) < self.epsilon:
                return self.action_space.sample()  # choose random action
        else:
                #import pdb; pdb.set_trace();
                if len(state) == 2:
                    state = state[0]
                state = torch.from_numpy(state).float()
               
                network_output_to_numpy = self.network(state).data.numpy()
                #print(np.argmax(network_output_to_numpy))
                return np.argmax(network_output_to_numpy)  # choose greedy action

    def update_Sarsa_Network(self, state, next_state, action, next_action, reward, terminals):

        try:
            qsa = torch.gather(self.network(state), dim=1, index=action.long())
        except:
            import pdb; pdb.set_trace();
        qsa_next_action = torch.gather(self.network(next_state), dim=1, index=next_action.long())

        not_terminals = 1 - terminals

        qsa_next_target = reward + not_terminals * (self.discount_factor * qsa_next_action)
        #import pdb; pdb.set_trace();
        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        self.optimizer.zero_grad()
        q_network_loss.backward()
        self.optimizer.step()
            
    def save(self):
        self.save_checkpoint(self.network, "ActionValue")

    def load(self, path):
        self.load_checkpoint(self.network, path, "ActionValue")
    
    def train(self, training_context: ReplayBuffer):
        for i in range(self.batch_size):
            states, next_states, actions, next_actions, rewards, terminals = training_context.random_sample_sarsa(64)
            states = torch.Tensor(states)
            next_states = torch.Tensor(next_states)
            actions = torch.Tensor(actions)
            next_actions = torch.Tensor(next_actions)
            rewards = torch.Tensor(rewards)
            terminals = torch.Tensor(terminals)
            self.update_Sarsa_Network(states, next_states, actions, next_actions, rewards, terminals)
            '''
                states, actions, rewards, next_states, next_actions, terminals = training_context.random_sample_sarsa(
            self.sample_size)
                #states, next_states, actions, next_actions, rewards, terminals = self.replay_buffer.sample_minibatch_sarsa(64)
                new_actions = []
                new_next_actions = []
                #import pdb; pdb.set_trace();
            
                for i in range(0, len(actions)):
                    new_actions.append([actions[i]])
                    new_next_actions.append([next_actions[i]])
                    #next_actions.append([self.get_action(next_states[i])])
                states = torch.Tensor(states)
                next_states = torch.Tensor(next_states)

                #next_actions = self.network(next_states)
                
                #print(np.argmax(network_output_to_numpy))
                #next_actions,_ = torch.max(next_actions, dim=1, keepdim=True)
                #import pdb; pdb.set_trace();
                
                actions = torch.Tensor(actions)
                next_actions = torch.Tensor(next_actions)
                rewards = torch.Tensor(rewards)
                terminals = torch.Tensor(terminals)
                self.update_Sarsa_Network(states, next_states, actions, next_actions, rewards, terminals)'''