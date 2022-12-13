import abc
import numpy as np


class AbstractNoise(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_action(self):
        raise NotImplementedError("not implemented")


class OUNoise(AbstractNoise):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        super().__init__()

        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0

        self.x_prev = None
        self.reset()

    def __call__(self):
        return self.get_action()

    def get_action(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
            self.mu, self.sigma)
