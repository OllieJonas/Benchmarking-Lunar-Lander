import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.max_cap = max_size
        self.input_dims = [8]
        self.mem_center = 0
        self.state_memory = np.zeros((self.max_cap, *self.input_dims))
        self.new_state_memory = np.zeros((self.max_cap, *self.input_dims))
        self.action_memory = np.zeros((self.max_cap, 2))
        self.reward_memory = np.zeros(self.max_cap)
        self.terminal_memory = np.zeros(self.max_cap, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_center % self.max_cap
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_center += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_center, self.max_cap)

        #batch = np.random.choice(max_mem, batch_size)
        batch = batch_size

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal
