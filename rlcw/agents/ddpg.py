import random
import numpy as np
from typing import NoReturn, List
from agents.abstract_agent import AbstractAgent

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from ..ddpg_models import *
from ..ddpg_utils import *

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

    