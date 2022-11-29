import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_capacity):
        self.max_capacity = max_capacity
        self.size = 0
        self.buffer = self._empty_buff()

    def add(self, item):
        if self.size >= self.max_capacity:
            self.buffer[0] = item
            self.buffer = np.roll(self.buffer, -1)
        else:
            self.buffer[self.size] = item
            self.size += 1

    def random_sample(self, sample_size):
        return np.random.choice(np.fromiter((x for x in self.buffer if x is not None),
                                            dtype=self.buffer.dtype), sample_size)

    def random_sample_transformed(self, sample_size, device):
        random_sample = self.random_sample(sample_size)

        # TODO: This is literal hot garbage. Please fix. Thanks! :)
        curr_states, next_states, rewards, actions, dones = [np.asarray(x) for x in zip(*random_sample)]

        curr_states = torch.from_numpy(curr_states).type(torch.FloatTensor).to(device)
        next_states = torch.from_numpy(next_states).type(torch.FloatTensor).to(device)
        rewards = torch.from_numpy(rewards).type(torch.FloatTensor).to(device)
        actions = torch.from_numpy(actions).type(torch.FloatTensor).to(device)
        dones = torch.from_numpy(dones).type(torch.FloatTensor).to(device)

        return curr_states, next_states, rewards, actions, dones

    def flush(self):
        self.buffer = self._empty_buff()
        self.size = 0

    def _empty_buff(self):
        return np.empty(self.max_capacity, dtype=object)

    def __getitem__(self, item):
        return self.buffer[item]

    def __repr__(self):
        return self.buffer.__repr__()
