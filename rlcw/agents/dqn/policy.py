import abc
import numpy as np
import torch


class AbstractPolicy(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self, action_values):
        pass


class EpsilonGreedyPolicy(AbstractPolicy):

    def __init__(self, epsilon: float, no_actions, device):
        super().__init__()
        self.epsilon = epsilon
        self.no_actions = no_actions
        self.device = device

    def get_action(self, action_values):
        if np.random.random() <= self.epsilon:  # random
            return np.random.randint(0, self.no_actions)
        else:  # greedy
            with torch.no_grad():
                return torch.argmax(action_values).item()

