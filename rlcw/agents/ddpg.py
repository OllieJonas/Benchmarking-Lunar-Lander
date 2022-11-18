import random
import numpy as np
from typing import NoReturn, List
from agents.abstract_agent import AbstractAgent

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ddpg_models import *
from ddpg_utils import *


class DdpgAgent(AbstractAgent):

    def __init__(self, logger, action_space, config):
        self.action_space = action_space
        self.Q = self._make_q(np.zeros(8))
        self.alpha = 0.1
        self.gamma = 0.9

        super().__init__(logger, action_space, config)

        self.num_states = action_space.shape[0]

    def name(self):
        return "ddpg"

    # returns the action for agent to do at beginning of each time-step
    def get_action(self, state):
        return super().get_action(state)

    # called when time-step to start training is reached
    def train(self, training_context: List) -> NoReturn:
        return super().train(training_context)


class ActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self. mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    # overrides call function, removes need for obj.meth(), can just use meth()
    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma + np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)
