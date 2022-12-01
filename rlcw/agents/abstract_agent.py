import torch

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @abstractmethod
    def get_action(self, state):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @abstractmethod
    def train(self, training_context: ReplayBuffer) -> NoReturn:
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)


class CheckpointAgent(AbstractAgent, ABC):

    def __init__(self, logger, action_space, config):
        super().__init__(logger, action_space, config)

    def save(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def load(self, path):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @staticmethod
    def _get_policy_path():
        return f"{util.get_curr_session_output_path()}policies/"

    @staticmethod
    def save_checkpoint(net, file_name):
        torch.save(net.state_dict(), util.with_file_extension(
            f"{CheckpointAgent._get_policy_path()}{file_name}", ".mdl"))

    @staticmethod
    def load_checkpoint(net, path, file_name):
        net.load_state_dict(torch.load(util.with_file_extension(
            f"{path}{file_name}", ".mdl")))

