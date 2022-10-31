import logging

from rlcw.abstract_agent import AbstractAgent
from rlcw.util import init_logger

LOGGER = init_logger(suffix="Main", level=logging.DEBUG)


def main():
    LOGGER.info("Hello world!")
    pass


def get_agent(name: str) -> AbstractAgent:
    pass


if __name__ == "__main__":
    main()
