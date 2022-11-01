import logging

from argparse import ArgumentParser, Namespace

import gym

from rlcw.agents.abstract_agent import AbstractAgent
from rlcw.agents.random import RandomAgent
from rlcw.util import init_logger

LOGGER = init_logger(suffix="Main")


def _make_env():
    return gym.make("LunarLander-v2", render_mode="human")


def main():
    # arg stuff
    args = _parse_cmd_line_args()
    LOGGER.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # create env
    env = _make_env()

    agent = get_agent(args.name, env.action_space)


def _parse_cmd_line_args() -> Namespace:
    arg_parser = ArgumentParser()

    arg_parser.add_argument("--name", type=str, help="Name of agent to use", default="random")
    arg_parser.add_argument("--verbose", action="store_true", help="Toggles debug printing", default=False)
    arg_parser.add_argument("--timesteps", type=int, help="Number of timesteps to use", default=1_000)
    arg_parser.add_argument("--render", action="store_true",
                            help="Toggles whether a UI should be rendered showing progress", default=True)

    return arg_parser.parse_args()


def get_agent(name: str, action_space) -> AbstractAgent:
    if name.lower() == "random":
        return RandomAgent(action_space)
    else:
        raise NotImplementedError("An agent of this name doesn't exist! :(")


class Runner:

    def __init__(self, env: gym.Env, agent: AbstractAgent, timesteps: int, render: bool, seed: int = 42):
        self.env = env
        self.agent = agent
        self.timesteps = timesteps
        self.render = render
        self.seed = seed

    def run(self):

        observation, info = self.env.reset()

        for t in range(self.timesteps):
            action = self.agent.get_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            if self.render:
                self.env.render()

            LOGGER.debug(f'Timestep {t}: Observation: {observation}, Reward: {reward}')

            if terminated or truncated:
                observation, info = self.env.reset()

        self.env.close()


if __name__ == "__main__":
    main()
