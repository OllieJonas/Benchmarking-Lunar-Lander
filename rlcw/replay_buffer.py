import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.mem_center = 0
        self.state_memory = np.zeros((self.mem_size, *[8]))
        self.new_state_memory = np.zeros((self.mem_size, *[8]))
        self.action_memory = np.zeros((self.mem_size, 2))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def add(self, item):
        index = self.mem_center % self.max_capacity
        self.state_memory[index] = item[0]
        self.new_state_memory[index] = item[1]
        self.reward_memory[index] = item[2]
        self.action_memory[index] = item[3]
        self.terminal_memory[index] = 1 - item[4]
        self.mem_center += 1

    def random_sample(self, batch_size):
        max_mem = min(self.mem_center, self.max_capacity)
        batch = np.random.choice(self.max_capacity, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def flush(self):
        self.buffer = self._empty_buff()
        self.size = 0

    def _empty_buff(self):
        return np.empty(self.max_capacity, dtype=object)

    def __getitem__(self, item):
        return self.buffer[item]

    def __repr__(self):
        return self.buffer.__repr__()
