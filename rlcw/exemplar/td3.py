import gym

from rlcw.util import init_logger
from rlcw.abstract_algorithm import AbstractAlgorithm

LOGGER = init_logger("TD3 (Exemplar)")


class TD3(AbstractAlgorithm):
    def train(self):
        LOGGER.info("Hello, world!")
