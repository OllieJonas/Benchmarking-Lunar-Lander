import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import rlcw.util as util


def save_plot_as_image(name, title, data, x_label, y_label):
    file_name = f'{util.get_curr_session_output_path()}results/png/{name}{"" if name.endswith(".png") else ".png"}'
    plt.plot(data)

    plt.ticklabel_format(style='plain', axis='x', useOffset=False)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(file_name)
    plt.close()


# noinspection PyTypeChecker
def save_as_csv(name, cumulative_rewards, average_rewards, no_timesteps):
    file_path = f'{util.get_curr_session_output_path()}results/csv/{name}{"" if name.endswith(".csv") else ".csv"}'

    cumulative_rewards = ["Cumulative Rewards"] + cumulative_rewards
    average_rewards = ["Average Rewards"] + average_rewards
    no_timesteps = ["No Timesteps"] + no_timesteps
    np.savetxt(file_path, [_ for _ in zip(cumulative_rewards, average_rewards, no_timesteps)], delimiter=', ', fmt="%s")


class Evaluator:
    def __init__(self, results, should_save_charts, should_save_csv, agent_name: str = ""):
        self.LOGGER = util.init_logger("Evaluator")

        self.results = results
        self.should_save_charts = should_save_charts
        self.should_save_csv = should_save_csv

        self.agent_name = agent_name

    def eval(self):
        self._eval_non_detailed()
        self._eval_detailed()
        return self.results

    def _eval_non_detailed(self):
        cumulative_rewards = [x[0] for x in self.results.results]
        average_rewards = [x[1] for x in self.results.results]
        no_timesteps = [x[2] for x in self.results.results]

        self.LOGGER.info(f'Cumulative Rewards: {cumulative_rewards}')
        self.LOGGER.info(f'Average Rewards: {average_rewards}')
        self.LOGGER.info(f'No Timesteps: {no_timesteps}')

        if self.should_save_charts:
            save_plot_as_image(f"cumulative_rewards ({self.agent_name}).png", "Cumulative Reward over each Episode",
                               cumulative_rewards, "Episode", "Reward")
            save_plot_as_image(f"average_rewards ({self.agent_name}).png", "Average Reward over each Episode",
                               average_rewards, "Episode", "Reward")
            save_plot_as_image(f"no_timesteps ({self.agent_name}).png", "Number of Timesteps",
                               no_timesteps, "Episode", "No Timesteps")

        if self.should_save_csv:
            save_as_csv("results.csv", cumulative_rewards, average_rewards, no_timesteps)

    def _eval_detailed(self):
        pass
