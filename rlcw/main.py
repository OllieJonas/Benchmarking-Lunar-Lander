import cProfile as profile
import logging
import pstats

import gym
import torch
import yaml

import logger
import util

from agents.ddpg.ddpg import DdpgAgent
from agents.dqn.dqn import DQN
from agents.deep_sarsa.deep_sarsa import DeepSarsaAgent
from agents.td3.td3 import Td3Agent
from agents.random import RandomAgent
from agents.sac.sac import SoftActorCritic
from agents.sarsa import SarsaAgent
from orchestrator import Orchestrator

LOGGER: logging.Logger


def _make_env(env_name, should_record, continuous, episodes_to_save):
    env = gym.make(env_name, continuous=continuous, render_mode="rgb_array") if should_record \
        else gym.make(env_name, continuous=continuous, render_mode="human")

    if should_record:
        env = gym.wrappers.RecordVideo(env, f'{util.get_curr_session_output_path()}results/recordings/',
                                       episode_trigger=lambda x: x in episodes_to_save)
    return env


def main():
    env, agent, config, episodes_to_save = setup()

    verbose = config["overall"]["output"]["verbose"]
    profiler = None

    if verbose:
        LOGGER.debug(f"Creating profiler...")
        profiler = profile.Profile()

    LOGGER.info(f'Marking episodes {episodes_to_save} for saving...')

    orchestrator = Orchestrator(
        env=env, agent=agent, config=config, episodes_to_save=episodes_to_save)
    orchestrator.load()

    if verbose:
        profiler.enable()

    orchestrator.run()

    if verbose:
        profiler.disable()
        stats = pstats.Stats(profiler).strip_dirs().sort_stats("cumtime")
        stats.dump_stats(f"{util.get_output_root_path()}logs/time.dmp")

    orchestrator.eval()


def get_agent(name: str, agents_config):
    """
    To add an agent, do the following template:
    elif name == "<your agents name">:
        return <Your Agent Class>(logger, action_space, cfg)
    """
    cfg = agents_config[name] if name in agents_config else None
    _logger = logger.init_logger(f'{name.upper()} (Agent)')
    name = name.lower()

    if name == "random":
        return RandomAgent(_logger, cfg)
    elif name == "sarsa":
        return SarsaAgent(_logger, cfg)
    elif name == "deep_sarsa":
        return DeepSarsaAgent(_logger, cfg)
    elif name == "ddpg":
        return DdpgAgent(_logger, cfg)
    elif name == "td3":
        return Td3Agent(_logger, cfg)
    elif name == "sac":
        return SoftActorCritic(_logger, cfg)
    elif name == "dqn":
        return DQN(_logger, cfg)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


def setup():
    global LOGGER

    config = _parse_config()

    config_overall = config["overall"]
    config_output = config_overall["output"]

    agent_name = config_overall["agent_name"]

    util.set_agent_name(agent_name)

    _make_dirs(config, agent_name)

    LOGGER = logger.init_logger("Main")

    logger_level = logging.DEBUG if config_output["verbose"] else logging.INFO

    LOGGER.setLevel(logger_level)
    logger.set_logger_level(logger_level)

    # can't render in human mode and record at the same time
    should_record = config_output["save"]["recordings"]
    should_render = config_output["render"]

    if should_render and should_record:
        LOGGER.warning(
            "Can't render and record at the same time! Disabling recording ...")
        should_record = False

    LOGGER.debug(f'Config: {config}')

    if not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available for Torch - Running on CPU ...")
    else:
        LOGGER.info("CUDA is enabled!")

    max_episodes = config_overall["episodes"]["max"]

    env_name = config_overall["env_name"]

    save_partitions = _parse_episode_config_var(
        max_episodes, config_output["save"]["episodes"])

    agent = get_agent(agent_name, config["agents"])

    env = _make_env(env_name, should_record,
                    agent.requires_continuous_action_space, save_partitions)

    agent.update_action_and_state_spaces(
        env.action_space, env.observation_space)
    agent.assign_env_dependent_variables(
        env.action_space, env.observation_space)

    return env, agent, config, save_partitions


def _make_dirs(config, agent_name):
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


def _parse_config(name="config.yml"):
    with open(f'{util.get_project_root_path()}{name}') as file:
        return yaml.safe_load(file)


def _parse_episode_config_var(max_episodes, inp):
    return list(map(lambda e: min(max_episodes - 1, e), inp)) if type(inp) is list else \
        [-1] if inp < 0 else _split_into_partitions(max_episodes, inp)


def _split_into_partitions(_max, partitions):
    """
    3 -> 0, 100, 199
    """
    if partitions <= 0:
        raise ValueError('partitions can\'t be less than 0')
    else:
        return list((min(_max - 1, i * _max // partitions) for i in range(partitions + 1)))


if __name__ == "__main__":
    main()
