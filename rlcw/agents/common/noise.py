import abc


class AbstractNoise(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def noise(self):
        raise NotImplementedError("not implemented")
