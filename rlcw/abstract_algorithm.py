import gym

class AbstractAlgorithm:

    def get_env(self):
        pass

    def run(self):
        raise NotImplementedError("This hasn't been implemented yet! :(")
