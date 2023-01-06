import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import util as util
import logger


def _get_csv_file_path(name):
    return f'{util.get_curr_session_output_path()}results/csv/{name}' \
           f'{"" if name.endswith(".csv") else ".csv"}'


def save_plot_as_image(name, title, data, x_label, y_label, fg_colour="red", bg_colour="grey", bg_opacity=0.35):
    file_name = util.with_file_extension(
        f'{util.get_curr_session_output_path()}results/png/{name}', "png")
    file_name = f'{util.get_curr_session_output_path()}results/png/{name}{"" if name.endswith(".png") else ".png"}'
    plt.plot(data)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.savefig(file_name)
    plt.close()


class Evaluator:
    def __init__(self, results, should_save_charts, should_save_csv, agent_name: str = ""):
        self.LOGGER = logger.init_logger("Evaluator")

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
            save_plot_as_image(f"cumulative_rewards.png", "Cumulative Reward over each Episode",
                               cumulative_rewards, "Episode", "Reward")
            save_plot_as_image(f"average_rewards.png", "Average Reward over each Episode",
                               average_rewards, "Episode", "Reward")
            save_plot_as_image(f"no_timesteps.png", "Number of Timesteps",
                               no_timesteps, "Episode", "No Timesteps")

        if self.should_save_csv:
            name = "results.csv"

            np.savetxt(_get_csv_file_path("results.csv"),
                       [_ for _ in zip(cumulative_rewards,
                                       average_rewards, no_timesteps)],
                       delimiter=', ',
                       fmt="%s")

    def _eval_detailed(self):
        for k, v in self.results.results_detailed.items():
            if self.should_save_charts:
                save_plot_as_image(f'rewards_ep_{str(k)}',
                                   f'Reward over Timesteps (Episode {str(k)})',
                                   [t.reward for t in v],
                                   "Timesteps",
                                   "Reward")

            if self.should_save_csv:
                name = f"results_ep_{str(k)}.csv"

                timesteps = ["Timestep"] + list(range(len(v)))
                states = ["State"] + [[s for s in t.state] for t in v]
                actions = ["Action"] + [t.action for t in v]
                rewards = ["Reward"] + [t.reward for t in v]

                np.savetxt(_get_csv_file_path(name),
                           [_ for _ in zip(timesteps, states,
                                           actions, rewards)],
                           delimiter=', ',
                           fmt="%s")
