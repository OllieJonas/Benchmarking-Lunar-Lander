import logging

import gym
import yaml

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from rlcw.orchestrator import Orchestrator
import rlcw.util as util

LOGGER: logging.Logger


def _make_env(env_name, should_record, episodes_to_save):
    env = gym.make(env_name, render_mode="rgb_array") if util.is_using_jupyter() or should_record \
        else gym.make(env_name, render_mode="human")

    if should_record:
        env = gym.wrappers.RecordVideo(env, f'{util.get_curr_session_output_path()}results/recordings/',
                                       episode_trigger=lambda x: x in episodes_to_save)
    return env


def enable_jupyter(value: bool = True):
    util.set_using_jupyter(value)


def main():
    env, agent, config, episodes_to_save = setup()
    LOGGER.info(f'Marking episodes {episodes_to_save} for saving...')
    orchestrator = Orchestrator(env=env, agent=agent, config=config, episodes_to_save=episodes_to_save)
    orchestrator.run()
    orchestrator.eval()


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
    _make_dirs(config)
    
    config_overall = config["overall"]
    LOGGER = util.init_logger("Main")

    logger_level = logging.DEBUG if config_overall["output"]["verbose"] else logging.INFO

    LOGGER.setLevel(logger_level)
    util.set_logger_level(logger_level)

    # can't render in human mode and record at the same time
    should_record = config_overall["output"]["save"]["recordings"]
    should_render = config_overall["output"]["render"]

    if should_render and should_record:
        LOGGER.warning("Can't render and record at the same time! Disabling recording ...")
        should_record = False

    LOGGER.debug(f'Config: {config}')

    max_episodes = config_overall["episodes"]["max"]
    no_episodes_to_save = config_overall["output"]["save"]["no_episodes"]

    env_name = config_overall["env_name"]

    save_partitions = _split_into_partitions(max_episodes, no_episodes_to_save)
    env = _make_env(env_name, should_record, save_partitions)

    agent = get_agent(config_overall["agent_name"], env.action_space, config["agents"])

    return env, agent, config, save_partitions


def _make_dirs(config):
    save_cfg = config["overall"]["output"]["save"]

    util.make_dir(util.get_output_root_path())

    session_path = util.get_curr_session_output_path()
    util.make_dir(session_path)

    policies_path = f'{session_path}policies/'
    util.make_dir(policies_path)

    results_path = f'{session_path}results/'
    png_path = f'{results_path}png/'
    raw_path = f'{results_path}raw/'
    csv_path = f'{results_path}csv/'
    recordings_path = f'{results_path}recordings/'

    util.make_dir(results_path)

    if save_cfg["charts"]:
        util.make_dir(png_path)

    if save_cfg["raw"]:
        util.make_dir(raw_path)

    if save_cfg["csv"]:
        util.make_dir(csv_path)

    if save_cfg["recordings"]:
        util.make_dir(recordings_path)


def _parse_config(path="../config.yml"):
    with open(path) as file:
        return yaml.safe_load(file)


def _split_into_partitions(_max, partitions):
    """
    3 -> 0, 100, 199
    """
    if partitions <= 0:
        raise ValueError('partitions can\'t be less than 0')
    else:
        return tuple((min(_max - 1, i * _max // partitions) for i in range(partitions + 1)))


if __name__ == "__main__":
    main()
