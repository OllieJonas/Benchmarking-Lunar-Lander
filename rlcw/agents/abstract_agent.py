import gym

from typing import NoReturn, List


class AbstractAgent:

    def __init__(self, logger, action_space, config):
        self.logger = logger
        self.action_space = action_space
        self.config = config

    def name(self):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def get_action(self, observation):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def train(self, training_context: List) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
