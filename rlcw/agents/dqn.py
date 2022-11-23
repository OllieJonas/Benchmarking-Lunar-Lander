from typing import NoReturn

import numpy as np
import torch as nn

from abstract_agent import CheckpointedAbstractAgent
from replay_buffer import ReplayBuffer


class DeepQNetwork(CheckpointedAbstractAgent):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)

    def save(self):
        pass

    def load(self):
        pass

    def name(self) -> str:
        pass

    def get_action(self, state):
        pass

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        pass
