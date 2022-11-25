import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import display

import util
from agents.abstract_agent import AbstractAgent
from replay_buffer import ReplayBuffer


class Orchestrator:

    def __init__(self, env, agent: AbstractAgent, config, episodes_to_save, seed: int = 42):
        self.LOGGER = util.init_logger("Orchestrator")

        self.env = env
        self.agent = agent
        self.config = config
        self.seed = seed
        self.episodes_to_save = episodes_to_save

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
        self.scores = self.runner.run()
        self.env.close()
        self.save_plot_as_image(f"average_rewards.png", "Average Reward over each Episode",
                                self.scores, "Episode", "Reward")

    def save_plot_as_image(self, name, title, data, x_label, y_label):

        print("\n\n")
        print(data)
        print("\n\n")
        print(len(data))
        num_eps = len(data)
        x = [i for i in range(num_eps)]

        file_name = f'{util.get_curr_session_output_path()}results/png/{name}{"" if name.endswith(".png") else ".png"}'

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, data)

        plt.savefig(file_name)
        plt.close()

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
                 max_episodes,
                 start_training_timesteps,
                 training_ctx_capacity):
        self.LOGGER = util.init_logger("Runner")

        self.env = env
        self.agent = agent
        self.seed = seed

        self.score_history = []

        self.plot_score = []

        self.episodes_to_save = episodes_to_save
        self.should_render = should_render
        self.max_episodes = max_episodes
        self.start_training_timesteps = start_training_timesteps
        self.training_ctx_capacity = training_ctx_capacity

    def run(self):
        # training_ctx_capacity = 64
        # max number of time-steps allowed in training_context
        training_context = ReplayBuffer(self.training_ctx_capacity)

        print(self.env.observation_space)

        curr_episode = 0

        for t in range(self.max_episodes):

            state, info = self.env.reset()
            done = False
            score = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)
                score += reward

                if terminated or truncated:
                    done = True

                # render
                if self.should_render:
                    self.env.render()

                training_context.add(np.array([
                    state,
                    next_state,
                    reward,
                    action,
                    done], dtype=object))

                if t > self.start_training_timesteps:
                    self.agent.train(training_context)

                state = next_state

                self.score_history.append(score)

            print('episode ', t, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % np.mean(self.score_history[-100:]))
            self.plot_score.append(score)

        return self.plot_score
