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

        self.scores = []
        self.score_history = []
        self.plot_score = []
        self.timesteps = []

        # eval stuff
        self.should_save_charts = _save_cfg["charts"]
        self.should_save_csv = _save_cfg["csv"]

        # self._sync_seeds()

    def run(self):
        self.LOGGER.info(f'Running agent {self.agent.name()} ...')
        self.run_agent()
        self.env.close()
        self.save_plot_as_image(f"average_rewards.png", "Average Reward over each Episode",
                                self.score_history, "Episode", "Reward")
        self.save_plot_as_image(f"past_100_eps_avg.png", "Average Reward over Past 100 Episodes",
                                self.plot_score, "Episode", "Reward")
        self.save_plot_as_image(f"average_timesteps.png", "Average Time-Steps over Past 100 Episodes",
                                self.timesteps, "Episode", "Time-Steps")

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

    def run_agent(self):
        agent_name = self.agent.name()
        if agent_name == "ddpg":
            self.run_ddpg()
        elif agent_name == "td3":
            self.run_td3()
        else:
            print("No agent with that name found")

    def run_td3(self):
        best_score = self.env.reward_range[0]
        self.score_history = []
        time_step_history = []

        self.agent.load_models()

        for t in range(self.max_episodes):
            state, info = self.env.reset()
            done = False
            score = 0
            time_steps = 0
            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)
                self.agent.store_memory(
                    state, action, reward, next_state, int(done))

                if terminated or truncated:
                    done = True

                # render
                if self.should_render:
                    self.env.render()

                self.agent.train()
                score += reward
                state = next_state
                time_steps += 1
            time_step_history.append(time_steps)
            self.timesteps.append(np.mean(time_step_history[-100:]))
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            print('episode ', t, 'score %.1f' % score,
                  'average score %.1f' % avg_score)

    def run_ddpg(self):
        time_step_history = []
        for t in range(self.max_episodes):

            state, info = self.env.reset()
            done = False
            score = 0
            time_steps = 0

            while not done:
                action = self.agent.get_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(
                    action)

                self.agent.store_memory(
                    state, action, reward, next_state, int(done))

                if terminated or truncated:
                    done = True

                # render
                if self.should_render:
                    self.env.render()

                self.agent.train()

                score += reward
                state = next_state
                time_steps += 1

            time_step_history.append(time_steps)
            self.timesteps.append(np.mean(time_step_history[-100:]))
            self.score_history.append(score)
            avg_score = np.mean(self.score_history[-100:])
            self.plot_score.append(avg_score)
            print('episode ', t, 'score %.2f' % score,
                  'trailing 100 games avg %.3f' % avg_score)
