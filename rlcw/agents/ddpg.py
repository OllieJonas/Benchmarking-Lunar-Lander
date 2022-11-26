import numpy as np
from typing import List

import torch as T
import torch.nn.functional as F

from actionNoise import ActionNoise
from criticNetwork import CriticNetwork
from actorNetwork import ActorNetwork
from replayBuffer import ReplayBuffer

# Code **inspired** by
# https://www.youtube.com/watch?v=6Yd5WnYls_Y


class DdpgAgent():

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

        self.memory = ReplayBuffer(self.max_size)

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  name='Actor')
        self.critic = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                    self.layer2_size, n_actions=self.n_actions,
                                    name='Critic')

        self.target_actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                         self.layer2_size, n_actions=self.n_actions,
                                         name='TargetActor')
        self.target_critic = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                           self.layer2_size, n_actions=self.n_actions,
                                           name='TargetCritic')

        self.noise = ActionNoise(mu=np.zeros(self.n_actions))

        self.update_network_parameters(tau=1)

    def save(self):
        pass

    def load(self):
        pass

    def name(self):
        return "ddpg"

    def store_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_action(self, observation):
        self.actor.eval()
        observation = T.tensor(
            observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def train(self, training_context):
        if self.memory.mem_center < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        done = T.tensor(done).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()
        target_actions = self.target_actor.forward(new_state)
        critic_value_ = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = T.tensor(target).to(self.critic.device)
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

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau * critic_state_dict[name].clone() + \
                (1 - tau) * target_critic_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                (1 - tau) * target_actor_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
