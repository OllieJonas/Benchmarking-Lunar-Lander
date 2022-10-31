from typing import NoReturn

from rlcw.agents.abstract_agent import AbstractAgent


class RandomAgent(AbstractAgent):

    def __init__(self, timesteps):
        super().__init__(timesteps)

    def get_action(self):
        pass

    def train(self) -> NoReturn:
        pass
