import gym

from typing import NoReturn


class AbstractAgent:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, observation):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def train(self) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
