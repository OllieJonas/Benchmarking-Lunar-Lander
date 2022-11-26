# code **inspired** by https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/TD3/td3_torch.py

import numpy as np
from typing import List

import torch as T
import torch.nn.functional as F

from actionNoise import ActionNoise
from criticNetwork import CriticNetwork
from actorNetwork import ActorNetwork
from replayBuffer import ReplayBuffer


class Td3Agent():

    def __init__(self, logger, action_space, config):
        self.action_space = action_space
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.input_dims = config['input_dims']
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']
        self.n_actions = config['n_actions']
        self.max_size = config['max_size']

        self.update_actor_interval = 2
        self.warmup = 1000
        self.max_action = self.action_space.high
        self.min_action = self.action_space.low
        self.learn_step_counter = 0
        self.time_step = 0

        self.memory = ReplayBuffer(self.max_size)

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  name='Actor')

        self.critic_one = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                        self.layer2_size, n_actions=self.n_actions,
                                        name='Critic_One')

        self.critic_two = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                        self.layer2_size, n_actions=self.n_actions,
                                        name='Critic_Two')

        self.target_actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                         self.layer2_size, n_actions=self.n_actions,
                                         name='TargetActor')

        self.target_critic_one = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                               self.layer2_size, n_actions=self.n_actions,
                                               name='TargetCritic_One')

        self.target_critic_two = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                               self.layer2_size, n_actions=self.n_actions,
                                               name='TargetCritic_Two')

        self.noise = ActionNoise(mu=np.zeros(self.n_actions))
        self.noise = config["noise"]

        self.update_network_parameters(tau=1)
