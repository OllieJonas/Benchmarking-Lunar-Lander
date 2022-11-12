import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import rlcw.util as util


def save_plot(name, title, data, x_label, y_label):
    file_name = f'{util.get_curr_session_output_path()}/results/png/{name}{"" if name.endswith(".png") else ".png"}'
    plt.plot(data)

    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(file_name)
    plt.close()


class Evaluator:
    def __init__(self, results, should_save_charts, should_save_csv):
        self.LOGGER = util.init_logger("Evaluator")

        self.results = results
        self.should_save_charts = should_save_charts
        self.should_save_csv = should_save_csv

    def eval(self):
        cumulative_rewards = [x[0] for x in self.results.results]
        average_rewards = [x[1] for x in self.results.results]
        no_timesteps = [x[2] for x in self.results.results]

        self.LOGGER.info(f'Cumulative Rewards: {cumulative_rewards}')
        self.LOGGER.info(f'Average Rewards: {average_rewards}')
        self.LOGGER.info(f'No Timesteps: {no_timesteps}')

        if self.should_save_charts:
            save_plot("cumulative_rewards", "Cumulative Reward over each Episode", cumulative_rewards,
                      "Episode", "Reward")
            save_plot("average_rewards", "Average Reward over each Episode", average_rewards,
                      "Episode", "Reward")
            save_plot("no_timesteps", "Number of Timesteps", no_timesteps, "Episode",
                      "No Timesteps")

        return self.results
