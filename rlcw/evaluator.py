import numpy as np
import matplotlib.pyplot as plt

import rlcw.util as util


def save_plot(name, title, data, x_label, y_label):
    file_name = f'{util.get_curr_session_output_path()}/results/{name}{"" if name.endswith(".png") else ".png"}'
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)


class Evaluator:
    def __init__(self, results):
        self.LOGGER = util.init_logger("Evaluator")

        self.results = results
        self.rewards_per_episodes = None

    def rewards_per_episode(self):
        if self.rewards_per_episodes is None:
            self.rewards_per_episodes = [np.fromiter(map(lambda t: t.reward, episode), dtype=float)
                                         for episode in self.results.results.values()]
        return self.rewards_per_episodes

    def rewards_ignoring_episodes(self):
        return np.concatenate(self.rewards_per_episode()).ravel()

    def cumulative_rewards_per_episode(self):
        return [np.sum(episode) for episode in self.rewards_per_episode()]

    def average_rewards_per_episode(self):
        return [np.average(episode) for episode in self.rewards_per_episode()]

    def no_timesteps_per_episode(self):
        return [episode.size for episode in self.rewards_per_episode()]

    def eval(self):
        self.LOGGER.debug(f'Raw: {self.results.results}')
        self.LOGGER.debug(f'Rewards (per Episode): {self.rewards_per_episode()}')
        self.LOGGER.debug(f'Rewards (ignoring Episodes): {self.rewards_ignoring_episodes()}')
        self.LOGGER.info(f'Cumulative Rewards (per Episode): {self.cumulative_rewards_per_episode()}')
        self.LOGGER.info(f'Average Rewards (per Episode): {self.average_rewards_per_episode()}')
        self.LOGGER.info(f'No Timesteps (per Episode): {self.no_timesteps_per_episode()}')
        return self.results
