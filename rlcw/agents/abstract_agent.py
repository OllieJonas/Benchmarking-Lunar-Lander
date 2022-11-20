from typing import NoReturn

from rlcw.replay_buffer import ReplayBuffer


class AbstractAgent:

    def __init__(self, logger, action_space, config):
        self.logger = logger
        self.action_space = action_space
        self.config = config

    def name(self) -> str:
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def get_action(self, state):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
