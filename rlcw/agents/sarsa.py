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

# neural network as q-table for lunar landing to too big
class ActionValueNetwork(nn.Module):
    def __init__(self, state_dim , action_dim):
        super(ActionValueNetwork, self).__init__()
        #import pdb; pdb.set_trace();
        self.x_layer = nn.Linear(state_dim.shape[0], 150)
        self.h_layer = nn.Linear(150, 120)
        self.y_layer = nn.Linear(120, action_dim.n)
        print(self.x_layer)

    def forward(self, state):
        xh = F.relu(self.x_layer(state))
        hh = F.relu(self.h_layer(xh))
        state_action_values = self.y_layer(hh)
        return state_action_values

class ReplayBuffer(object):
    def __init__(self):
        self.buffer = []
        self.buffer_s = []
        
    def add_to_buffer(self, data):
        #data must be of the form (state,next_state,action,reward,terminal)
        self.buffer.append(data)
        
    def add_to_buffer_sarsa(self, data):
        #data must be of the form (state,next_state,action,n_action,reward,terminal)
        self.buffer_s.append(data)

    def sample_minibatch(self,minibatch_length):
        states = []
        next_states = []
        actions = []
        rewards = []
        terminals = []
        for i in range(minibatch_length):
            random_int = np.random.randint(0, len(self.buffer)-1) 
            transition = self.buffer[random_int]
            states.append(transition[0])
            next_states.append(transition[1])
            actions.append(transition[2])
            rewards.append(transition[3])
            terminals.append(transition[4])
        return torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(rewards), torch.Tensor(terminals)

    def sample_minibatch_sarsa(self,minibatch_length):
        
        states = []
        next_states = []
        actions = []
        next_actions = []
        rewards = []
        terminals = []
        for i in range(minibatch_length):
            
            random_int = np.random.randint(0, len(self.buffer_s)-1) 
            transition = self.buffer_s[64]
            '''
            states = np.append(states, transition[0])
            next_states = np.append(next_states, transition[1])
            actions = np.append(actions, transition[2])
            next_actions = np.append(next_actions, transition[3])
            rewards = np.append(rewards, transition[4])
            terminals = np.append(terminals, transition[5])
            '''
            states.append(transition[0])
            next_states.append(transition[1])
            actions.append(transition[2])
            next_actions.append(transition[3])
            rewards.append(transition[4])
            terminals.append(transition[5])
            
        #states = np.squeeze(states)
        #next_states = np.squeeze(next_states)
        #print("hi")
        #print(states)
        '''
        print("next states")
        print(next_states)
        #print(torch.Tensor(states))
        print("action")
        print(actions)
        print("next_actions")
        print(next_actions)
        print("reward")
        print(reward)
        print("terminal")states, next_states, actions, next_actions, reward, terminals
        print(terminal)'''
        # this corrects the array randomly being considered an object
        for i in range(0, len(states)):
            if len(states[i]) == 2:
                states[i] = states[i][0]
        '''
        try:
            torch.Tensor(states)
        except:
            #print([states[0]])
            #print([next_states[0]])
            if len(states[0]) == 2:
                states[0] = states[0][0]
            #print(len(states[0]))
            return torch.Tensor([states[0]]), torch.Tensor([next_states[0]]), torch.Tensor([actions[0]]), torch.Tensor([next_actions[0]]), torch.Tensor([rewards[0]]), torch.Tensor([terminals[0]])
         '''   
        
        return torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(next_actions), torch.Tensor(rewards), torch.Tensor(terminals)#torch.from_numpy(states), torch.from_numpy(next_states), torch.from_numpy(actions), torch.from_numpy(next_actions), torch.from_numpy(rewards), torch.from_numpy(terminals)# torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(next_actions), torch.Tensor(rewards), torch.Tensor(terminals)

# Technically Deep/expected SARSA but t'is late so just SARSA

class SarsaAgent(CheckpointAgent):

    def name(self):
        return "sarsa"
    
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.criterion = nn.MSELoss()

        self.num_replay = 8

        # config vars
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.sample_size = config["sample_size"]
        self.batch_size = config["batch_size"]
        self.epsilon = 0.99

        self.lr = 0.001

        self.loss = 0

        self.replay_buffer = ReplayBuffer()
        self.num_actions = None
        self.optimizer = None
        self.network = None
        self.discount_factor = 0.99
        self.MSELoss_function = nn.MSELoss()
        self.replay_buffer = ReplayBuffer()

    def assign_env_dependent_variables(self, action_space, state_space):
        self.num_actions = action_space.n

        self.network = ActionValueNetwork(state_space, action_space).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

    def decay_epsilon(self):
        if self.epsilon > 0.2:
            self.epsilon *= 0.996 #change this so eplison does not decay as fast
    
        if self.epsilon <= 0.2:
            self.epsilon = 0.2

    def get_action(self, state):

        if np.random.uniform(0, 1) < self.epsilon:
                return self.action_space.sample()  # choose random action
        else:
                #import pdb; pdb.set_trace();
                
                state = torch.from_numpy(state).float()
                network_output_to_numpy = self.network(state).data.numpy()
                return np.argmax(network_output_to_numpy)  # choose greedy action
    
    def update_Sarsa_Network(self, state, next_state, action, next_action, reward, terminals):

        qsa = torch.gather(self.network(state), dim=1, index=action.long())

        qsa_next_action = torch.gather(self.network(next_state), dim=1, index=next_action.long())

        not_terminals = 1 - terminals

        qsa_next_target = reward + not_terminals * (self.discount_factor * qsa_next_action)

        q_network_loss = self.MSELoss_function(qsa, qsa_next_target.detach())
        self.optimizer.zero_grad()
        q_network_loss.backward()
        self.optimizer.step()
            
    def save(self):
        self.save_checkpoint(self.network, "ActionValue")

    def load(self, path):
        self.load_checkpoint(self.network, path, "ActionValue")
    
    def train(self, training_context: ReplayBuffer):
        
        if training_context.cnt < self.batch_size:
            return
        else:
            for i in range(self.batch_size):
                states, actions, rewards, next_states, terminals = training_context.random_sample(
            self.sample_size)
                #states, next_states, actions, next_actions, rewards, terminals = self.replay_buffer.sample_minibatch_sarsa(64)
                new_actions = []
                next_actions = []
                for i in range(0, len(states)):
                    if len(states[i]) == 2:
                        states[i] = states[i][0]
                    new_actions.append([actions[i]])
                    next_actions.append([self.get_action(next_states[i])])

                states = torch.Tensor(states)
                next_states = torch.Tensor(next_states)
                actions = torch.Tensor(new_actions)
                next_actions = torch.Tensor(next_actions)
                rewards = torch.Tensor(rewards)
                terminals = torch.Tensor(terminals)
                self.update_Sarsa_Network(states, next_states, actions, next_actions, rewards, terminals)

    def best_move(self, state):
        
        return np.argmax(self.network(state).data.numpy())
