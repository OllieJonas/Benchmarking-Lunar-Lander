import logging

import gym
import yaml

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from util import init_logger, make_dir

LOGGER = init_logger(suffix="Main")


def _make_env():
    return gym.make("LunarLander-v2", render_mode="human")


def main():
    # arg stuff
    config = _parse_config()
    _make_dirs()

    LOGGER.setLevel(logging.DEBUG if config["verbose"] else logging.INFO)
    LOGGER.debug(f'Config: {config}')
    LOGGER.info("Hello, world!")

    env = _make_env()
    agent = get_agent(config["agent"], env.action_space)
    runner = Runner(env=env, agent=agent, config=config)
    runner.run()


def get_agent(name: str, action_space) -> AbstractAgent:
    if name.lower() == "random":
        return RandomAgent(action_space)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


class Runner:

    def __init__(self, env: gym.Env, agent: AbstractAgent, config, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed
        self.results = Results()

    def run(self):

        observation, info = self.env.reset()
        training_context = []

        for t in range(self.config["timesteps"]["total"]):
            action = self.agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)

            if self.config["render"]:
                self.env.render()

            training_context.append({
                "curr_obsv": observation,
                "next_obsv": next_observation,
                "reward": reward,
                "action": action
            })

            if t > self.config["timesteps"]["start_training"]:
                self.agent.train(training_context)

            next_observation = observation

            result_obj = Results.ResultObj(timestamp=t, observation=observation, reward=reward)

            self.results.add(result_obj)
            LOGGER.debug(result_obj)

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()


class Eval:

    def __init__(self, results):
        self.results = results


class Results:
    class ResultObj:
        def __init__(self, timestamp, observation, reward):
            self.timestamp = timestamp
            self.observation = observation
            self.reward = reward

        def __str__(self):
            return f'Timestep {self.timestamp}: Observation: {self.observation}, Reward: {self.reward}'

    def __init__(self):
        self._results = []

    def add(self, result: ResultObj):
        self._results.append(result)

    def save(self, file_name):
        pass


def _make_dirs():
    root_path = "../"
    results_path = f'{root_path}results'
    policies_path = f'{root_path}policies'

    make_dir(results_path, logger=LOGGER)
    make_dir(policies_path, logger=LOGGER)


def _parse_config():
    with open("../config.yml") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
