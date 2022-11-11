import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython import display

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.util import init_logger


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, from_jupyter: bool = False, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.from_jupyter = from_jupyter
        self.seed = seed

        self.runner = None
        self.eval = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.config, seed=self.seed)
        self.runner.run(from_jupyter=self.from_jupyter)
        self.env.close()

    def eval(self):
        pass

    def _sync_seeds(self):
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)


class Runner:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.LOGGER = init_logger("Runner")
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed
        self.results = Results()

    def run(self, from_jupyter: bool = False):
        observation, info = self.env.reset()
        training_context = []

        max_episodes = self.config["episodes"]["max"]
        episode_count = 1

        # display
        image = plt.imshow(self.env.render()) if from_jupyter else None

        for t in range(self.config["timesteps"]["total"]):
            if episode_count > max_episodes:
                break

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
            self.LOGGER.debug(result_obj)

            if terminated or truncated:
                observation, info = self.env.reset()


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
