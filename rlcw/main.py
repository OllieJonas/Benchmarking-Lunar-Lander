import logging

import gym
import yaml
import matplotlib
import matplotlib.pyplot as plt

from IPython import display

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from rlcw.util import init_logger, make_dir

LOGGER = init_logger(suffix="Main")


def _make_env(from_jupyter=False):
    return gym.make("LunarLander-v2", render_mode="rgb_array") if from_jupyter \
        else gym.make("LunarLander-v2", render_mode="human")


def main():
    env, agent, config = setup()
    runner = Runner(env=env, agent=agent, config=config)
    runner.run()


def setup(from_jupyter: bool = False):
    config = _parse_config("../../config.yml" if from_jupyter else "../config.yml")
    _make_dirs()

    LOGGER.setLevel(logging.DEBUG if config["verbose"] else logging.INFO)
    LOGGER.debug(f'Config: {config}')

    env = _make_env(from_jupyter=from_jupyter)
    agent = get_agent(config["agent"], env.action_space)

    return env, agent, config


def get_agent(name: str, action_space) -> AbstractAgent:
    if name.lower() == "random":
        return RandomAgent(action_space)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, from_jupyter: bool = False, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.from_jupyter = from_jupyter
        self.seed = seed

        self.runner = None
        self.eval = None

    def run(self):
        self.runner = Runner(self.env, self.agent, self.config, seed=self.seed)
        self.runner.run(from_jupyter=self.from_jupyter)

    def eval(self):
        pass


class Runner:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed
        self.results = Results()

    def run(self, from_jupyter: bool = False):
        observation, info = self.env.reset()
        training_context = []

        # display
        image = plt.imshow(self.env.render()) if from_jupyter else None
        for t in range(self.config["timesteps"]["total"]):
            action = self.agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)

            # render
            if self.config["render"]:
                if from_jupyter:
                    image.set_data(self.env.render())
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                else:
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


def _parse_config(path="../config.yml"):
    with open(path) as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
