import logging

from argparse import ArgumentParser, Namespace

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.util import init_logger

LOGGER = init_logger(suffix="Main")


def main():
    args = parse_cmd_line_args()

    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    LOGGER.info("Hello world!")


def parse_cmd_line_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--verbose", action="store_true", help="Toggles debug printing", default=False)
    arg_parser.add_argument("--timesteps", type=int, help="Number of timesteps to use", default=1_000)

    return arg_parser.parse_args()


def get_agent(name: str) -> AbstractAgent:
    pass


if __name__ == "__main__":
    main()
