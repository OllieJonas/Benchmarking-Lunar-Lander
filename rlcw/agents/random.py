import random

from typing import NoReturn, List

from rlcw.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, action_space):
        super().__init__(action_space)

    def get_action(self, observation):
        return random.choice(self.action_space.n)

    def train(self, training_context: List) -> NoReturn:
        pass
