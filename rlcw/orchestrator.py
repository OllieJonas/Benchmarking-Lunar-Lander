import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython import display

import rlcw.util as util

from rlcw.evaluator import Evaluator
from rlcw.agents.abstract_agent import AbstractAgent


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed

        self.runner = None
        self.results = None

        self.evaluator = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.config, seed=self.seed)
        self.results = self.runner.run()
        self.env.close()

        self.results.save_to_disk()

    def eval(self):
        self.evaluator = Evaluator(self.results)

    def _sync_seeds(self):
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)


class Runner:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.LOGGER = util.init_logger("Runner")
        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed
        self.results = Results(agent_name=agent.name(), date_time=util.CURR_DATE_TIME)

    def run(self):
        state, info = self.env.reset()
        training_context = []

        max_episodes = self.config["episodes"]["max"]
        curr_episode = 0

        is_using_jupyter = util.is_using_jupyter()

        # display
        image = plt.imshow(self.env.render()) if is_using_jupyter else None

        for t in range(self.config["timesteps"]["total"]):
            if curr_episode > max_episodes:
                break

            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # render
            if self.config["render"]:
                if is_using_jupyter:
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
            result_obj = Results.ResultObj(episode=curr_episode, timestep=t, state=state, reward=reward)

            self.results.add(curr_episode, result_obj)
            self.LOGGER.debug(result_obj)

            if terminated:
                curr_episode += 1
                state, info = self.env.reset()

            if truncated:
                state, info = self.env.reset()

        return self.results


class Results:
    class ResultObj:
        def __init__(self, episode, timestep, state, reward):
            self.episode = episode
            self.timestep = timestep
            self.state = state
            self.reward = reward

        def __str__(self):
            return f'Timestep {self.timestep}: State: {self.state}, Reward: {self.reward}'

    def __init__(self, agent_name, date_time):
        self.agent_name = agent_name
        self.date_time = date_time

        self._results = []

    def add(self, episode: int, result: ResultObj):
        if len(self._results) != episode + 1:
            self._results.append([])

        self._results[episode].append(result)

    def save_to_disk(self):
        file_name = f'{self.agent_name} - {self.date_time}'
        util.save_file("results", file_name, self._results)
