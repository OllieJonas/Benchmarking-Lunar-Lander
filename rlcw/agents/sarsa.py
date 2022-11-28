import random

import numpy as np

from typing import NoReturn, List

from agents.abstract_agent import AbstractAgent
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
import util


DEVICE = util.get_torch_device()

# neural network as q-table for lunar landing to too big
class ActionValueNetwork(nn.Module):
   
    def __init__(self, observation, action_space):
        super().__init__()
        
        self.state_dim = observation.shape[0]
        self.num_hidden_units =  256
        self.num_actions = action_space.n

        self.hidden_layer = nn.Linear(self.state_dim, self.num_hidden_units)
        self.output_layer = nn.Linear(self.num_hidden_units, self.num_actions)
                
    
    def forward(self, state):
        """
        This is a feed-forward pass in the network
        Args:
            s (Numpy array): The state, a 2D array of shape (batch_size, state_dim)
        Returns:
            The action-values (Torch array) calculated using the network's weights.
            A 2D array of shape (batch_size, num_actions)
        """
        state = torch.Tensor(state)
        
        q_vals = F.relu(self.hidden_layer(state))
        q_vals = self.output_layer(q_vals)

        return q_vals
class ReplayBuffer(object):
    def __init__(self, max_size, mini_batch):
        self.minibatch_size = mini_batch
        self.buffer = [] 
        self.max_size = max_size
        self.input_dims = [8]
        self.mem_center = 0
        self.state_memory = np.zeros((self.max_size, *self.input_dims))
        self.new_state_memory = np.zeros((self.max_size, *self.input_dims))
        self.action_memory = np.zeros((self.max_size, 2))
        self.reward_memory = np.zeros(self.max_size)
        self.terminal_memory = np.zeros(self.max_size, dtype=np.float32)

    def append(self, state, action, reward, terminal, next_state):
        index = self.mem_center % self.max_size
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = terminal
        self.mem_center += 1

    def sample(self):
        max_mem = min(self.mem_center, self.max_size)
        batch = np.random.choice(max_mem, self.minibatch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def size(self):
        return len(self.buffer)

# Technically Deep/expected SARSA
class SarsaAgent():
    def name(self):
        return "sarsa"

    def __init__(self, logger, action_space, observation_space, config):
        """
        Set parameters needed to setup the agent.
        """
        self.observation_space =  observation_space
        self.action_space = action_space

        self.replay_buffer = ReplayBuffer(64, 1)

        self.network = ActionValueNetwork(observation_space, action_space).to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr = 1e-3, 
                                    betas=(0.9, 0.999),
                                    eps=1e-8) 
        self.criterion = nn.MSELoss()
        self.num_actions = action_space.n
        self.num_replay = 4
        self.discount = 0.99
        self.tau = 0.001
                
        self.last_state = None
        self.last_action = None
        
        self.sum_rewards = 0
        self.episode_steps = 0
        self.loss = 0

    def get_action(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        action_values = self.network.forward(state)
        probs_batch = self.softmax(action_values, self.tau).detach().numpy()
        action = np.random.choice(self.num_actions, p=probs_batch.squeeze())
        return action

    def agent_start(self, state):
        """The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's evn_start function.
        Returns:
            The first action the agent takes.
        """
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state])
        self.last_action = self.get_action(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.array([state])

        action = self.get_action(state) 
        self.replay_buffer.append(self.last_state, self.last_action, reward, 0, state)
        
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):                
                experiences = self.replay_buffer.sample()
                
                self.loss +=self.optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau,
                                 self.criterion)
                
        self.last_state = state
        self.last_action = action
        
        return self.last_action

    def agent_end(self, reward):
        """Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        
        state = np.zeros_like(self.last_state)

        self.replay_buffer.append(self.last_state, self.last_action, reward, 1, state)
        
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                
                experiences = self.replay_buffer.sample()
                
                self.loss += self.optimize_network(experiences, self.discount, self.optimizer, self.network, current_q, self.tau,
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
        sum_of_exp_preferences = torch.sum(exp_preferences, dim = 1)
        
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

        if terminals:
            terminals = [1]
        else:
            terminals = [0]
        v_next_vec = torch.sum(probs_mat * q_next_mat, dim = 0) * (1 - torch.tensor(terminals))    

        target_vec = torch.tensor(rewards) + (discount * v_next_vec)

        q_mat = network.forward(states)
        
        batch_indices = torch.arange(q_mat.shape[0])

        estimate_vec = q_mat[batch_indices, actions]  

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
    
        states, actions, rewards, terminals, next_states = map(list, zip(*experiences))

        states = np.concatenate(states) 
        next_states = np.concatenate(next_states)
        rewards = np.array(rewards) 
        terminals = np.array(terminals) 
        batch_size = states.shape[0] 
        
        td_target, td_estimate = self.get_td(states, next_states, actions, rewards, discount, terminals, \
                                            network, current_q, tau)
        
        # zero the gradients buffer
        optimizer.zero_grad()
        loss = criterion(td_estimate.double().to(DEVICE), td_target.to(DEVICE))

        loss.backward()

        optimizer.step()
        
        return (loss / batch_size).detach().numpy()
