import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython import display

from rlcw.main import USING_JUPYTER
from rlcw.evaluator import Evaluator
from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.util import init_logger


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed

        self.runner = None
        self.evaluator = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.config, seed=self.seed)
        self.runner.run()
        self.env.close()

    def eval(self):
        self.evaluator = Evaluator()

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

    def run(self):
        state, info = self.env.reset()
        training_context = []

        max_episodes = self.config["episodes"]["max"]
        curr_episode = 0

        # display
        image = plt.imshow(self.env.render()) if USING_JUPYTER else None

        for t in range(self.config["timesteps"]["total"]):
            if curr_episode > max_episodes:
                break

            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # render
            if self.config["render"]:
                if USING_JUPYTER:
                    image.set_data(self.env.render())
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                else:
                    self.env.render()

            training_context.append({
                "curr_state": state,
                "next_state": next_state,
                "reward": reward,
                "action": action
            })

            if t > self.config["timesteps"]["start_training"]:
                self.agent.train(training_context)

            next_state = state
            result_obj = Results.ResultObj(timestep=t, state=state, reward=reward)

            self.results.add(curr_episode, result_obj)
            self.LOGGER.debug(result_obj)

            if terminated:
                curr_episode += 1
                state, info = self.env.reset()

            if truncated:
                state, info = self.env.reset()


class Results:
    class ResultObj:
        def __init__(self, episode, timestep, state, reward):
            self.episode = episode
            self.timestep = timestep
            self.state = state
            self.reward = reward

        def __str__(self):
            return f'Timestep {self.timestep}: State: {self.state}, Reward: {self.reward}'

    def __init__(self):
        self._results = []

    def add(self, episode: int, result: ResultObj):
        if not self._results[episode]:
            self._results[episode] = []

        self._results[episode].append(result)

    def save(self, file_name):
        pass
