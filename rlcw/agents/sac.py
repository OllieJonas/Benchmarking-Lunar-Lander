from typing import NoReturn

from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer


class SoftActorCritic(AbstractAgent):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)
        self.logger.info(f'SAC Config: {config}')

    def name(self) -> str:
        return "SAC"

    def get_action(self, state):
        pass

    def train(self, training_context: ReplayBuffer) -> NoReturn:
        pass
