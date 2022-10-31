import gym

from typing import NoReturn


def _make_env():
    return gym.make("LunarLander-v2", render_mode="human")


class AbstractAlgorithm:

    def __init__(self):
        env = _make_env()

    def train(self) -> NoReturn:
        raise NotImplementedError("This hasn't been implemented yet! :(")
