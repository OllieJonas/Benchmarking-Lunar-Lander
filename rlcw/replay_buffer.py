import numpy as np


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

    def flush(self):
        self.buffer = self._empty_buff()
        self.size = 0

    def _empty_buff(self):
        return np.empty(self.max_capacity, dtype=object)

    def __getitem__(self, item):
        return self.buffer[item]

    def __repr__(self):
        return self.buffer.__repr__()
