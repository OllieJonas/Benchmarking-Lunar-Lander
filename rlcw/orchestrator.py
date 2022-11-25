import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display

import util
from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, seed: int = 42):
        self.LOGGER = util.init_logger("Orchestrator")

        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed

        _save_cfg = config["overall"]["output"]["save"]

        # runner stuff
        self.should_render = config["overall"]["output"]["render"]
        self.should_save_raw = _save_cfg["raw"]

        self.max_episodes = config["overall"]["episodes"]["max"]
        self.start_training_timesteps = config["overall"]["timesteps"]["start_training"]
        self.training_ctx_capacity = config["overall"]["context_capacity"]

        self.runner = None
        self.time_taken = 0.
        self.results = None

        self.scores = []

        # eval stuff
        self.should_save_charts = _save_cfg["charts"]
        self.should_save_csv = _save_cfg["csv"]

        self.evaluator = None

        self._sync_seeds()

    def run(self):
        self.runner = Runner(self.env, self.agent, self.seed,
                             episodes_to_save=self.episodes_to_save,
                             should_render=self.should_render,
                             max_episodes=self.max_episodes,
                             start_training_timesteps=self.start_training_timesteps,
                             training_ctx_capacity=self.training_ctx_capacity)

        self.LOGGER.info(f'Running agent {self.agent.name()} ...')
        self.results, self.scores = self.runner.run()
        # self.time_taken = end - start
        # self.LOGGER.info(f'Time Taken: {self.time_taken}')
        self.env.close()

        if self.should_save_raw:
            self.results.save_to_disk()

    def eval(self):
        self.evaluator = eval.Evaluator(self.scores, self.results, self.should_save_charts, self.should_save_csv,
                                        agent_name=self.agent.name())
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
                 start_training_timesteps,
                 training_ctx_capacity):
        self.LOGGER = util.init_logger("Runner")

        self.env = env
        self.agent = agent
        self.seed = seed

        self.score = []

        self.score_history = []

        self.episodes_to_save = episodes_to_save
        self.should_render = should_render
        self.max_timesteps = max_timesteps
        self.max_episodes = max_episodes
        self.start_training_timesteps = start_training_timesteps
        self.training_ctx_capacity = training_ctx_capacity

        self.results = Results(agent_name=agent.name(),
                               date_time=util.CURR_DATE_TIME)

    def run(self):
        # training_ctx_capacity = 64
        # max number of time-steps allowed in training_context
        training_context = ReplayBuffer(self.training_ctx_capacity)

        print(self.env.observation_space)

        curr_episode = 0

        # display
        image = plt.imshow(self.env.render())

        for t in range(self.max_episodes):

            state, info = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)
                score += reward

                # render
                if self.should_render:
                    self.env.render()

                training_context.add(np.array([
                    state,
                    next_state,
                    reward,
                    action,
                    terminated], dtype=object))

                if t > self.start_training_timesteps:
                    self.agent.train(training_context)

                state = next_state

                # my score info
                self.score_history.append(score)

                timestep_result = Results.Timestep(
                    state=state, action=action, reward=reward)
                summary = self.results.add(
                    curr_episode, timestep_result, curr_episode in self.episodes_to_save)

        return self.results, self.score


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

    def save_score(self, reward):
        self.score.append(reward)

    def add(self, episode: int, timestep: Timestep, store_detailed: bool):
        if episode == self.curr_episode:
            self.timestep_buffer.append(timestep)
            return None
        else:
            if store_detailed:
                self.results_detailed[episode] = [t.clone()
                                                  for t in self.timestep_buffer]

            self.curr_episode = episode

            rewards = np.fromiter(
                map(lambda t: t.reward, self.timestep_buffer), dtype=float)
            cumulative = np.sum(rewards)
            avg = np.average(rewards)
            no_timesteps = rewards.size

            episode_summary = (cumulative, avg, no_timesteps)
            self.results.append(episode_summary)
            # flush buffer
            self.timestep_buffer = []

            return episode_summary

    def save_to_disk(self):
        file_name = f'{self.agent_name} - {self.date_time}'
        util.save_file("results", file_name, self.results.__str__())
