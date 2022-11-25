import random

from typing import NoReturn, List


class RandomAgent():

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)
        self.logger.info(
            f'I\'ve read my config file and found the value "{config["foo"]}" for foo!')
        self.logger.info(f'Here\'s my entire config file: {config}')

    def name(self):
        return "Random"

    def get_action(self, state):
        return random.choice(range(self.action_space.n))

    def train(self, training_context: List) -> NoReturn:
        pass
