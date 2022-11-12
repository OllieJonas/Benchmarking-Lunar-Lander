import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from IPython import display

import rlcw.util as util

import rlcw.evaluator as eval
from rlcw.agents.abstract_agent import AbstractAgent


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, episodes_to_save, seed: int = 42):
        self.LOGGER = util.init_logger("Orchestrator")

        self.env = env
        self.agent = agent
        self.config = config
        self.episodes_to_save = episodes_to_save
        self.seed = seed

        _save_cfg = config["overall"]["output"]["save"]

        # runner stuff
        self.should_render = config["overall"]["output"]["render"]
        self.should_save_raw = _save_cfg["raw"]

        self.max_episodes = config["overall"]["episodes"]["max"]

        self.max_timesteps = config["overall"]["timesteps"]["max"]
        self.start_training_timesteps = config["overall"]["timesteps"]["start_training"]

        self.runner = None
        self.time_taken = 0.
        self.results = None

        # eval stuff
        self.should_save_charts = _save_cfg["charts"]
        self.should_save_csv = _save_cfg["csv"]

        self.evaluator = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.seed,
                             episodes_to_save=self.episodes_to_save,
                             should_render=self.should_render,
                             max_timesteps=self.max_timesteps,
                             max_episodes=self.max_episodes,
                             start_training_timesteps=self.start_training_timesteps)

        self.LOGGER.info(f'Running agent {self.agent.name()} ...')
        self.results = self.runner.run()
        # self.time_taken = end - start
        self.LOGGER.info(f'Time Taken: {self.time_taken}')
        self.env.close()

        if self.should_save_raw:
            self.results.save_to_disk()

    def eval(self):
        self.evaluator = eval.Evaluator(self.results)
        self.evaluator.eval()

    def _sync_seeds(self):
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)


class Runner:

    def __init__(self, env, agent: AbstractAgent, seed: int,
                 should_render,
                 episodes_to_save,
                 max_timesteps,
                 max_episodes,
                 start_training_timesteps):
        self.LOGGER = util.init_logger("Runner")

        self.env = env
        self.agent = agent
        self.seed = seed

        self.episodes_to_save = episodes_to_save
        self.should_render = should_render
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
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

            state = next_state

            if curr_episode in self.episodes_to_save:
                result_obj = Results.Timestep(state=state, action=action, reward=reward)
                self.results.add(curr_episode, result_obj)
                self.LOGGER.debug(result_obj)

            if terminated:
                curr_episode += 1
                state, info = self.env.reset()

            if truncated:
                state, info = self.env.reset()

        return self.results


class Results:
    """
    idk how this is going to interact with pytorch cuda parallel stuff so maybe we'll have to forget this? atm,
    this is responsible for recording results.

    """
    class Timestep:
        def __init__(self, state, action, reward):
            self.state = state
            self.action = action
            self.reward = reward

        def __repr__(self):
            return f'<s: {self.state}, a: {self.action}, r: {self.reward}>'

        def clone(self):
            return Results.Timestep(self.state, self.action, self.reward)

    def __init__(self, agent_name, date_time):
        self.agent_name = agent_name
        self.date_time = date_time

        self.timestep_buffer = []
        self.curr_episode = 0

        self.results = []

        self.results_detailed = {}

    def __repr__(self):
        return self.results.__str__()

    def add(self, episode: int, timestep: Timestep, store_detailed: bool):
        if episode == self.curr_episode:
            self.timestep_buffer.append(timestep)
        else:
            if store_detailed:
                self.results_detailed[episode] = [t.clone() for t in self.timestep_buffer]

            print(self.timestep_buffer)
            # flush buffer
            self.timestep_buffer = []

    def save_to_disk(self):
        file_name = f'{self.agent_name} - {self.date_time}'
        util.save_file("results", file_name, self.results.__str__())
