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

        # getting necessary stuff out of config
        self.should_render = config["overall"]["output"]["render"]
        self.should_save_timesteps = config["overall"]["output"]["save_timesteps"]

        self.max_episodes = config["overall"]["episodes"]["max"]

        self.max_timesteps = config["overall"]["timesteps"]["max"]
        self.start_training_timesteps = config["overall"]["timesteps"]["start_training"]

        self.runner = None
        self.results = None

        self.evaluator = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.seed,
                             should_render=self.should_render,
                             max_timesteps=self.max_timesteps,
                             max_episodes=self.max_episodes,
                             start_training_timesteps=self.start_training_timesteps)

        self.results = self.runner.run()
        self.env.close()

        if self.should_save_timesteps:
            self.results.save_to_disk()

    def eval(self):
        self.evaluator = Evaluator(self.results)

    def _sync_seeds(self):
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)


class Runner:

    def __init__(self, env, agent: AbstractAgent, seed: int,
                 should_render,
                 max_timesteps,
                 max_episodes,
                 start_training_timesteps):
        self.LOGGER = util.init_logger("Runner")

        self.env = env
        self.agent = agent
        self.seed = seed

        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.should_render = should_render
        self.start_training_timesteps = start_training_timesteps

        self.results = Results(agent_name=agent.name(), date_time=util.CURR_DATE_TIME)

    def run(self):
        state, info = self.env.reset()
        training_context = []

        curr_episode = 0

        is_using_jupyter = util.is_using_jupyter()

        # display
        image = plt.imshow(self.env.render()) if is_using_jupyter else None

        for t in range(self.max_timesteps):
            if curr_episode > self.max_episodes:
                break

            action = self.agent.get_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)

            # render
            if self.should_render:
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

            if t > self.start_training_timesteps:
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

        def __repr__(self):
            return f'<t {self.timestep}: s: {self.state}, r: {self.reward}>'

    def __init__(self, agent_name, date_time):
        self.agent_name = agent_name
        self.date_time = date_time

        self.results = []

    def __repr__(self):
        return self.results.__str__()

    def add(self, episode: int, result: ResultObj):
        if len(self.results) != episode + 1:
            self.results.append([])

        self.results[episode].append(result)

    def save_to_disk(self):
        file_name = f'{self.agent_name} - {self.date_time}'
        util.save_file("results", file_name, self.results.__str__())
