import numpy as np

from rlcw.orchestrator import Results
import rlcw.util as util
import matplotlib.pyplot as plt


def main():
    state = []
    # state = [1.1154175e-03, 1.4161105e+00, 1.1296810e-01, 2.3067845e-01
    #          - 1.2857395e-03, -2.5588963e-02, 0.0000000e+00, 0.0000000e+00]

    results = Results("random", util.CURR_DATE_TIME)
    results.add(0, Results.ResultObj(0, 0, state, 3))
    results.add(1, Results.ResultObj(0, 1, state, 5))
    results.add(2, Results.ResultObj(0, 2, state, 4))
    results.add(3, Results.ResultObj(0, 3, state, 9))
    results.add(4, Results.ResultObj(0, 4, state, 8))
    results.add(5, Results.ResultObj(0, 5, state, 2))
    results.add(6, Results.ResultObj(0, 6, state, 12))

    results = results.results
    reward_list_per_episode = [np.fromiter(map(lambda t: t.reward, episode), dtype=float) for episode in results]

    cumulative_rewards = [np.sum(i) for i in reward_list_per_episode]
    avg_rewards = [np.average(i) for i in reward_list_per_episode]

    print(reward_list_per_episode)
    print(rewards_ignoring_episodes)
    print(avg_rewards)
    print(cumulative_rewards)


if __name__ == "__main__":
    main()
