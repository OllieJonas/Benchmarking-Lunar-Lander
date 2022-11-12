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
        self.results = results
        self.rewards_per_episodes = None

    def rewards_per_episode(self):
        if self.rewards_per_episodes is None:
            self.rewards_per_episodes = [np.fromiter(map(lambda t: t.reward, episode), dtype=float)
                                         for episode in self.results.results]
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
        return self.results
