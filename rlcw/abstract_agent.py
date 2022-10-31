import gym

from typing import NoReturn


def _make_env():
    return gym.make("LunarLander-v2", render_mode="human")


class AbstractAgent:

    def __init__(self, timesteps: int):
        self.timesteps = timesteps
        self.env = _make_env()

    def get_action(self):
        raise NotImplementedError("This hasn't been implemented yet! :(")

    def train(self) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
