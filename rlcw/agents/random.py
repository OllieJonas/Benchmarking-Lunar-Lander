import random

from typing import NoReturn, List

import util
from rlcw.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, action_space, config):
        super().__init__(action_space, config)
        logger = util.init_logger("Random")
        logger.info(f'I\'ve read my config file and found the value "{config["foo"]}" for foo!')

    def name(self):
        return "Random"

    def get_action(self, observation):
        return random.choice(range(self.action_space.n))

    def train(self, training_context: List) -> NoReturn:
        pass
