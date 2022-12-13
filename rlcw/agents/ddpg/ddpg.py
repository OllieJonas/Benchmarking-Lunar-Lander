"""
author: Fraser
"""
import numpy as np
import torch as T
import torch.nn.functional as F

import agents.common.utils as agent_utils
from agents.abstract_agent import CheckpointAgent
from agents.common.noise import OUNoise
from agents.ddpg.networks import ActorNetwork, CriticNetwork


class DdpgAgent(CheckpointAgent):
    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.requires_continuous_action_space = True

        # config vars
        self.alpha = config['alpha']
        self.beta = config['beta']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.batch_size = config['batch_size']
        self.layer1_size = config['layer1_size']
        self.layer2_size = config['layer2_size']

        self.no_actions = None
        self.input_dims = None
        self.noise = None

        self.actor, self.actor_optim = None, None
        self.critic, self.critic_optim = None, None
        self.target_actor = None
        self.target_critic = None

    def assign_env_dependent_variables(self, action_space, state_space):
        self.input_dims = state_space.shape
        self.no_actions = action_space.shape[0]

        self.actor, self.actor_optim = agent_utils.with_optim(ActorNetwork(self.input_dims, self.layer1_size,
                                                                           self.layer2_size, no_actions=self.no_actions,
                                                                           ), self.alpha, device=self.device)

        self.critic, self.critic_optim = agent_utils.with_optim(CriticNetwork(self.input_dims, self.layer1_size,
                                                                              self.layer2_size,
                                                                              no_actions=self.no_actions,
                                                                              ), self.beta, device=self.device)

        self.target_actor = ActorNetwork(self.input_dims, self.layer1_size,
                                         self.layer2_size, no_actions=self.no_actions,
                                         ).to(self.device)

        self.target_critic = CriticNetwork(self.input_dims, self.layer1_size,
                                           self.layer2_size, no_actions=self.no_actions,
                                           ).to(self.device)

        self.noise = OUNoise(mu=np.zeros(self.no_actions))

        agent_utils.hard_copy(self.actor, self.target_actor)
        agent_utils.hard_copy(self.critic, self.target_critic)

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
        self.critic_optim.zero_grad(set_to_none=True)
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic_optim.step()
        self.critic.eval()

        self.actor_optim.zero_grad(set_to_none=True)

        mu = self.actor.forward(state)
        self.actor.train()

        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()

        self.actor_optim.step()

        agent_utils.soft_copy(self.actor, self.target_actor, self.tau)
        agent_utils.soft_copy(self.critic, self.target_critic, self.tau)

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
