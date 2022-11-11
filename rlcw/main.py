import logging

import gym
import yaml

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from rlcw.orchestrator import Orchestrator
import rlcw.util as util

LOGGER: logging.Logger


def _make_env():
    return gym.make("LunarLander-v2", render_mode="rgb_array") if util.is_using_jupyter() \
        else gym.make("LunarLander-v2", render_mode="human")


def enable_jupyter(value: bool = True):
    util.set_using_jupyter(value)


def main():
    env, agent, config = setup()

    orchestrator = Orchestrator(env=env, agent=agent, config=config)
    orchestrator.run()


def get_agent(name: str, action_space, config) -> AbstractAgent:
    name = name.lower()
    cfg = config[name] if name in config else None

    if name == "random":
        return RandomAgent(action_space, cfg)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


def is_using_jupyter():
    return util.is_using_jupyter()


def setup():
    global LOGGER

    config = _parse_config("../../config.yml" if util.is_using_jupyter() else "../config.yml")
    _make_dirs()

    LOGGER = util.init_logger("Main")

    logger_level = logging.DEBUG if config["overall"]["output"]["verbose"] else logging.INFO

    LOGGER.setLevel(logger_level)
    util.set_logger_level(logger_level)

    LOGGER.debug(f'Config: {config}')

    env = _make_env()
    agent = get_agent(config["overall"]["agent_name"], env.action_space, config["agents"])

    return env, agent, config


def _make_dirs():
    root_path = util.get_root_output_path()
    util.make_dir(root_path)

    results_path = f'{root_path}results'
    policies_path = f'{root_path}policies'

    util.make_dir(results_path)
    util.make_dir(policies_path)


def _parse_config(path="../config.yml"):
    with open(path) as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
