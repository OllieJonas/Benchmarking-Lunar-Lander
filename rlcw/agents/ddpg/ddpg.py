"""
author: Fraser
"""
import numpy as np
import torch as T
import torch.nn.functional as F
from torch import optim

from agents.abstract_agent import CheckpointAgent
from agents.ddpg.action_noise import OUNoise
from agents.ddpg.networks import ActorNetwork, CriticNetwork


class DdpgAgent(CheckpointAgent):
    def __init__(self, logger, action_space, state_space, config):
        super().__init__(logger, action_space, config)
        self.action_space = action_space
        self.state_space = state_space
        self.input_dims = state_space.shape
        self.no_actions = action_space.shape[0]

        # config vars
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']

        self.actor = ActorNetwork(self.input_dims, self.layer1_size,
                                  self.layer2_size, no_actions=self.no_actions,
                                  ).to(self.device)

        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.alpha)

        self.critic = CriticNetwork(self.input_dims, self.layer1_size,
                                    self.layer2_size, no_actions=self.no_actions,
                                    ).to(self.device)

        self.critic_optimiser = optim.Adam(self.critic.parameters(), lr=self.beta)

        self.target_actor = ActorNetwork(self.input_dims, self.layer1_size,
                                         self.layer2_size, no_actions=self.no_actions,
                                         ).to(self.device)

        self.target_critic = CriticNetwork(self.input_dims, self.layer1_size,
                                           self.layer2_size, no_actions=self.no_actions,
                                           ).to(self.device)

        self.noise = OUNoise(mu=np.zeros(self.no_actions))

        self._hard_copy(self.actor, self.target_actor)
        self._hard_copy(self.critic, self.target_critic)

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
        self.critic_optimiser.zero_grad(set_to_none=True)
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optimiser.step()
        self.critic.eval()

        self.actor_optimiser.zero_grad(set_to_none=True)

        mu = self.actor.forward(state)
        self.actor.train()

        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()

        self.actor_optimiser.step()

        self._soft_copy(self.actor, self.target_actor)
        self._soft_copy(self.critic, self.target_critic)

    def _soft_copy(self, from_net, to_net):
        state_dict = dict(from_net.named_parameters())
        target_state_dict = dict(to_net.named_parameters())

        for _n in state_dict:
            state_dict[_n] = self.tau * state_dict[_n].clone() + (1 - self.tau) * target_state_dict[_n].clone()

        to_net.load_state_dict(state_dict)

    @staticmethod
    def _hard_copy(from_net, to_net):
        state_dict = dict(from_net.named_parameters())
        target_state_dict = dict(to_net.named_parameters())

        for _n in state_dict:
            state_dict[_n] = target_state_dict[_n].clone()

        to_net.load_state_dict(state_dict)

    def save(self):
        self.save_checkpoint(self.actor, "Actor")
        self.save_checkpoint(self.critic, "Critic")
        self.save_checkpoint(self.target_actor, "TargetActor")
        self.save_checkpoint(self.target_critic, "TargetCritic")

    def load(self, path):
        self.load_checkpoint(self.actor, path, "Actor")
        self.load_checkpoint(self.critic, path, "Critic")
        self.load_checkpoint(self.target_actor, path, "TargetActor")
        self.load_checkpoint(self.target_critic, path, "TargetCritic")
