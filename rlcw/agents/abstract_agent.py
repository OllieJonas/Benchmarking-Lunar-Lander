from typing import NoReturn
from abc import abstractmethod, ABC

import util
from rlcw.replay_buffer import ReplayBuffer

NOT_IMPLEMENTED_MESSAGE = "This hasn't been implemented yet! :("


class AbstractAgent(ABC):

    def __init__(self, logger, action_space, config):
        self.logger = logger
        self.action_space = action_space
        self.config = config

        self.device = util.get_torch_device()

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @abstractmethod
    def train(self, training_context: ReplayBuffer) -> NoReturn:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class CheckpointedAbstractAgent(AbstractAgent, ABC):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)

    def save(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def load(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def save_checkpoint(self, net):
        pass

    def load_checkpoint(self, net):
        pass

