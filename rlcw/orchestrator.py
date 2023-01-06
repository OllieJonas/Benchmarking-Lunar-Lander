import copy
import random

import numpy as np
import torch

import evaluator as eval
import logger
import util
from agents.abstract_agent import AbstractAgent, CheckpointAgent
from runners import Runner


class Orchestrator:

    def __init__(self, env, agent, config, episodes_to_save, seed: int = 42):
        self.LOGGER = logger.init_logger("Orchestrator")

        self.env = env
        self.agent = agent

        self.agent_config = config["agents"][agent.name().lower()]
        self.config = config

        self.episodes_to_save = episodes_to_save
        self.seed = seed

        self.verbose = config["overall"]["output"]["verbose"]

        # checkpoint stuff
        self.should_save_checkpoints = config["overall"]["checkpoint"]["save"]["enabled"]
        self.save_every = config["overall"]["checkpoint"]["save"]["every"]

        self.should_load_from_checkpoint = config["overall"]["checkpoint"]["load"]["enabled"]
        self.should_use_latest_run_for_load = config["overall"]["checkpoint"]["load"]["use_latest_run"]
        self.should_use_relative_path = config["overall"]["checkpoint"]["load"]["custom"]["use_relative"]
        self.load_directory = config["overall"]["checkpoint"]["load"]["custom"]["path"]

        # runner stuff
        self.should_render = config["overall"]["output"]["render"]

        self.max_episodes = config["overall"]["episodes"]["max"]

        self.max_timesteps = config["overall"]["timesteps"]["max"]
        self.max_ep_timestep = config["overall"]["timesteps"]["episode_timesteps"]
        self.start_training_timesteps = config["overall"]["timesteps"]["start_training"]
        self.training_ctx_capacity = config["overall"]["context_capacity"]

        self.should_invert_done = self.agent_config.get("invert_done", True)

        self.time_taken = 0.
        self.results = None

        # eval stuff
        _save_cfg = config["overall"]["output"]["save"]

        self.should_save_raw = _save_cfg["raw"]
        self.should_save_charts = _save_cfg["charts"]
        self.should_save_csv = _save_cfg["csv"]

        self.evaluator = None

        self._sync_seeds()

    def load(self):
        loader = Loader(enabled=self.should_load_from_checkpoint,
                        agent_name=self.agent.name(),
                        use_latest=self.should_use_latest_run_for_load,
                        use_relative=self.should_use_relative_path,
                        path=self.load_directory)

        loader.load(self.agent)

    def run(self):
        runner = Runner(self.env, self.agent, self.seed,
                        episodes_to_save=self.episodes_to_save,
                        should_render=self.should_render,
                        max_timesteps=self.max_timesteps,
                        max_ep_timestep=self.max_ep_timestep,
                        max_episodes=self.max_episodes,
                        start_training_timesteps=self.start_training_timesteps,
                        training_ctx_capacity=self.training_ctx_capacity,
                        should_save_checkpoints=self.should_save_checkpoints,
                        save_every=self.save_every,
                        should_invert_done=self.should_invert_done,
                        verbose=self.verbose)

        self.LOGGER.info(f'Running agent {self.agent.name()} ...')
        self.results = runner.run_agent()
        # self.time_taken = end - start
        # self.LOGGER.info(f'Time Taken: {self.time_taken}')
        self.env.close()

        if self.should_save_raw:
            self.results.save_to_disk()

    def eval(self):
        self.evaluator = eval.Evaluator(self.results, self.should_save_charts, self.should_save_csv,
                                        agent_name=self.agent.name())
        self.evaluator.eval()

    def _sync_seeds(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.random.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)


class Loader:
    def __init__(self, enabled, agent_name, use_relative, use_latest, path):
        self.LOGGER = logger.init_logger("CheckpointLoader")
        self.is_enabled = enabled
        self.agent_name = agent_name
        self.use_relative = use_relative
        self.use_latest = use_latest
        self.path = path

    def load(self, agent):
        if self.is_enabled:
            if not isinstance(agent, CheckpointAgent):
                self.LOGGER.warning(
                    "Can't load checkpoints for this agent! Disabling...")
            else:
                path = self._get_path()
                self.LOGGER.info(f"Loading enabled! Loading from {path}...")
                agent.load(self.path)

    def _get_path(self):
        latest_policies = self._get_latest_policies_for(self.agent_name)

        if latest_policies is None and self.use_latest:
            self.LOGGER.critical(
                f"Can't find a run for agent with name {self.agent_name}! Shutting down ...")
            exit(1)

        return latest_policies if self.use_latest \
            else f"{util.get_project_root_path() if self.use_relative else '/'}{self.path}"

    @staticmethod
    def _get_latest_policies_for(name: str):
        latest = util.get_latest_run_of(name)

        if not latest:
            return None

        return latest
