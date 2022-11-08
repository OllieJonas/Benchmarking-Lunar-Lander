import logging

import gym
import yaml

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from rlcw.orchestrator import Orchestrator
from rlcw.util import init_logger, make_dir, set_logger_level

LOGGER = init_logger(suffix="Main")


def _make_env(from_jupyter=False):
    return gym.make("LunarLander-v2", render_mode="rgb_array") if from_jupyter \
        else gym.make("LunarLander-v2", render_mode="human")


def main():
    env, agent, config = setup()

    orchestrator = Orchestrator(env=env, agent=agent, config=config)
    orchestrator.run()


def get_agent(name: str, action_space) -> AbstractAgent:
    if name.lower() == "random":
        return RandomAgent(action_space)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


def setup(from_jupyter: bool = False):
    config = _parse_config("../../config.yml" if from_jupyter else "../config.yml")
    _make_dirs()

    logger_level = logging.DEBUG if config["verbose"] else logging.INFO

    LOGGER.setLevel(logger_level)
    set_logger_level(logger_level)

    LOGGER.debug(f'Config: {config}')

    env = _make_env(from_jupyter=from_jupyter)
    agent = get_agent(config["agent"], env.action_space)

    return env, agent, config


def _make_dirs():
    root_path = "../"
    results_path = f'{root_path}results'
    policies_path = f'{root_path}policies'

    make_dir(results_path, logger=LOGGER)
    make_dir(policies_path, logger=LOGGER)


def _parse_config(path="../config.yml"):
    with open(path) as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
