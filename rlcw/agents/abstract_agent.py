import gym

from typing import NoReturn, List


class AbstractAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def name(self):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def get_action(self, observation):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def train(self, training_context: List) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
