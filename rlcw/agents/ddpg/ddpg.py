"""
author: Fraser
"""
import numpy as np
from typing import List

import torch as T
import torch.nn.functional as F

from agents.ddpg.action_noise import ActionNoise
from agents.ddpg.networks import ActorNetwork, CriticNetwork

from agents.abstract_agent import CheckpointAgent


class DdpgAgent(CheckpointAgent):

    def __init__(self, logger, action_space, state_space, config):
        super().__init__(logger, action_space, config)
        self.action_space = action_space
        self.state_space = state_space
        self.input_dims = state_space.shape

        # config vars
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']
        self.n_actions = config['n_actions']
        self.max_size = config['max_size']

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  ).to(self.device)
        self.critic = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                    self.layer2_size, n_actions=self.n_actions,
                                    ).to(self.device)

        self.target_actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                         self.layer2_size, n_actions=self.n_actions,
                                         ).to(self.device)
        self.target_critic = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                           self.layer2_size, n_actions=self.n_actions,
                                           ).to(self.device)

        self.noise = ActionNoise(mu=np.zeros(self.n_actions))

        self.update_network_parameters(tau=1)

    def name(self):
        return "ddpg"

    def get_action(self, observation):
        self.actor.eval()
        observation = T.tensor(
            observation, dtype=T.float).to(self.device)
        mu = self.actor.forward(observation).to(self.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def train(self, training_context):
        if training_context.cnt < self.batch_size:
            return
        else:
            self._do_train(training_context)

    def _do_train(self, training_context):

        state, action, reward, new_state, done = training_context.random_sample_as_tensors(
            self.batch_size, self.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []

        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * target_critic_value[j] * done[j])
        target = T.tensor(target).to(self.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimiser.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimiser.step()

        self.critic.eval()
        self.actor.optimiser.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimiser.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        def _soft_copy(state_dict, target_state_dict, target, tau):
            for _n in state_dict:
                state_dict[_n] = tau * state_dict[_n].clone() + (1 - tau) * target_state_dict[_n].clone()

            target.load_state_dict(state_dict)

        _soft_copy(dict(self.critic.named_parameters()), dict(self.target_critic.named_parameters()),
                   self.target_critic, tau)

        _soft_copy(dict(self.actor.named_parameters()), dict(self.target_actor.named_parameters()),
                   self.target_actor, tau)

    def load(self, path):
        self.load_checkpoint(self.actor, path, "Actor")
        self.load_checkpoint(self.critic, path, "Critic")
        self.load_checkpoint(self.target_actor, path, "TargetActor")
        self.load_checkpoint(self.target_critic, path, "TargetCritic")

    def save(self):
        self.save_checkpoint(self.actor, "Actor")
        self.save_checkpoint(self.critic, "Critic")
        self.save_checkpoint(self.target_actor, "TargetActor")
        self.save_checkpoint(self.target_critic, "TargetCritic")
