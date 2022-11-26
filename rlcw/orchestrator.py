import matplotlib.pyplot as plt
import numpy as np
import torch

import util


class Orchestrator:

    def __init__(self, env, agent, config, episodes_to_save, seed: int = 42):
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
        self.max_timesteps = config["overall"]["timesteps"]["max"]

        self.max_episodes = config["overall"]["episodes"]["max"]
        self.start_training_timesteps = config["overall"]["timesteps"]["start_training"]
        self.training_ctx_capacity = config["overall"]["context_capacity"]

        self.runner = None
        self.time_taken = 0.
        self.results = None

        self.scores = []
        self.score_history = []
        self.plot_score = []

        # eval stuff
        self.should_save_charts = _save_cfg["charts"]
        self.should_save_csv = _save_cfg["csv"]

        # self._sync_seeds()

    def run(self):
        self.LOGGER.info(f'Running agent {self.agent.name()} ...')
        self.scores = self.agent_run()
        self.env.close()
        self.save_plot_as_image(f"average_rewards.png", "Average Reward over each Episode",
                                self.scores, "Episode", "Reward")
        self.save_plot_as_image(f"past_100_eps_avg.png", "Average Reward over Passed 100 Episodes",
                                self.plot_score, "Episode", "Reward")

    def save_plot_as_image(self, name, title, data, x_label, y_label):
        num_eps = len(data)
        x = [i for i in range(num_eps)]

        file_name = f'{util.get_curr_session_output_path()}results/png/{name}{"" if name.endswith(".png") else ".png"}'

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.plot(x, data)

        plt.savefig(file_name)
        plt.close()

    def _sync_seeds(self):
        np.random.seed(self.seed)
        torch.random.manual_seed(self.seed)

    def agent_run(self):
        for t in range(self.max_episodes):

            state, info = self.env.reset()
            done = False
            score = 0

            name = self.agent.name()

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)

                if name == "ddpg":
                    self.agent.store_memory(
                        state, action, reward, next_state, int(done))

                if terminated or truncated:
                    done = True

                # render
                if self.should_render:
                    self.env.render()

                self.agent.train(None)

                score += reward
                state = next_state

            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            self.plot_score.append(avg_score)
            print('episode ', t, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % avg_score)

        return self.score_history
