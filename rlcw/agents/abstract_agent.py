import torch

from typing import NoReturn
from abc import abstractmethod, ABC

import util

from replay_buffer import ReplayBuffer

NOT_IMPLEMENTED_MESSAGE = "This hasn't been implemented yet! :("


class AbstractAgent(ABC):

    def __init__(self, logger, config):
        self.logger = logger
        self.config = config

        self.action_space = None
        self.state_space = None

        self.requires_continuous_action_space = False

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def update_action_and_state_spaces(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

    def assign_env_dependent_variables(self, action_space, state_space):
        pass

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

    def __init__(self, logger, config):
        super().__init__(logger, config)

    def save(self):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    def load(self, path):
        raise NotImplementedError(NOT_IMPLEMENTED_MESSAGE)

    @staticmethod
    def save_checkpoint(net, file_name):
        path = util.with_file_extension(
            f"{util.get_curr_session_output_path()}policies/{file_name}", ".pth")
        torch.save(net.state_dict(), path)

    @staticmethod
    def load_checkpoint(net, path, file_name):
        net.load_state_dict(torch.load(util.with_file_extension(
            f"{path}{file_name}", ".pth")))
