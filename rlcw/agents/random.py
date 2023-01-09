import random

from typing import NoReturn, List

from agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.logger.info(
            f'I\'ve read my config file and found the value "{config["foo"]}" for foo!')
        self.logger.info(f'Here\'s my entire config file: {config}')

    def name(self):
        return "Random"

    def assign_env_dependent_variables(self, action_space, state_space):
        pass

    def get_action(self, state):
        return random.choice(range(self.action_space.n))

    def train(self, training_context: List) -> NoReturn:
        pass
