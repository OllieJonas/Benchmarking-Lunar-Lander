import numpy as np

import util


class Results:
    """
    idk how this is going to interact with pytorch cuda parallel stuff, so maybe we'll have to forget this? atm,
    this is responsible for recording results.
    """

    class Timestep:
        def __init__(self, state, action, reward):
            self.state = state
            self.action = action
            self.reward = reward

        def __repr__(self):
            return f'<s: {self.state}, a: {self.action}, r: {self.reward}>'

        def clone(self):
            return Results.Timestep(self.state, self.action, self.reward)

    def __init__(self, agent_name, date_time):
        self.agent_name = agent_name
        self.date_time = date_time

        self.timestep_buffer = []
        self.curr_episode = 0

        self.results = []

        self.results_detailed = {}

    def __repr__(self):
        return self.results.__str__()

    def add(self, episode: int, timestep: Timestep, store_detailed: bool):
        if episode == self.curr_episode:
            self.timestep_buffer.append(timestep)
            return None
        else:
            if store_detailed:
                self.results_detailed[episode] = [t.clone() for t in self.timestep_buffer]

            self.curr_episode = episode

            rewards = np.fromiter(map(lambda t: t.reward, self.timestep_buffer), dtype=float)
            cumulative = np.sum(rewards)
            avg = np.average(rewards)
            no_timesteps = rewards.size

            episode_summary = (cumulative, avg, no_timesteps)
            self.results.append(episode_summary)
            # flush buffer
            self.timestep_buffer = []

            return episode_summary

    def save_to_disk(self):
        file_name = f'{self.agent_name} - {self.date_time}'
        util.save_file("results", file_name, self.results.__str__())