import numpy as np
import torch as T
import torch.nn.functional as F

import agents.common.utils as agent_utils
from agents.abstract_agent import CheckpointAgent
from agents.td3.networks import ActorNetwork
from agents.td3.networks import CriticNetwork


class Td3Agent(CheckpointAgent):

    def __init__(self, logger, config):
        super().__init__(logger, config)

        self.requires_continuous_action_space = True

        # config vars
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
        self.noise = config["noise"]

        self.update_actor_interval = 2
        self.warmup = 1000

        self.learn_step_counter = 0
        self.time_step = 0

        self.target_critic_two, self.target_critic_two_optim = None, None
        self.target_critic_one, self.target_critic_one_optim = None, None
        self.target_actor, self.target_actor_optim = None, None
        self.critic_two, self.critic_two_optim = None, None
        self.critic_one, self.critic_one_optim = None, None
        self.actor_two, self.actor_two_optim = None, None
        self.actor_one, self.actor_one_optim = None, None
        self.min_action, self.max_action = None, None

    def assign_env_dependent_variables(self, action_space, state_space):
        # self.min_action, self.max_action = action_space.low, action_space.high
        self.min_action, self.max_action = action_space.low, action_space.high

        self.actor_one, self.actor_one_optim = agent_utils.with_optim(
            ActorNetwork(self.input_dims, self.layer1_size,
                         self.layer2_size,
                         self.n_actions),
            self.alpha, device=self.device)

        self.actor_two, self.actor_two_optim = agent_utils.with_optim(
            ActorNetwork(self.input_dims, self.layer1_size,
                         self.layer2_size, self.n_actions),
            self.alpha, device=self.device)

        self.critic_one, self.critic_one_optim = agent_utils.with_optim(
            CriticNetwork(self.input_dims, self.layer1_size,
                          self.layer2_size,
                          no_actions=self.n_actions),
            self.beta, device=self.device)

        self.critic_two, self.critic_two_optim = agent_utils.with_optim(
            CriticNetwork(self.input_dims, self.layer1_size,
                          self.layer2_size,
                          no_actions=self.n_actions),
            self.beta, device=self.device)

        self.target_actor, self.target_actor_optim = agent_utils.with_optim(
            ActorNetwork(self.input_dims, self.layer1_size,
                         self.layer2_size, no_actions=self.n_actions), self.alpha)

        self.target_critic_one, self.target_critic_one_optim = agent_utils.with_optim(
            CriticNetwork(self.input_dims, self.layer1_size,
                          self.layer2_size, no_actions=self.n_actions), self.beta, device=self.device)

        self.target_critic_two, self.target_critic_two_optim = agent_utils.with_optim(
            CriticNetwork(self.input_dims, self.layer1_size,
                          self.layer2_size, no_actions=self.n_actions), self.beta, device=self.device)

        agent_utils.hard_copy(self.critic_one, self.target_critic_one)
        agent_utils.hard_copy(self.critic_two, self.target_critic_two)
        agent_utils.hard_copy(self.actor_one, self.target_actor)

    def save(self):
        self.save_checkpoint(self.actor_one, "ActorOne")
        self.save_checkpoint(self.actor_two, "ActorTwo")
        self.save_checkpoint(self.critic_one, "CriticOne")
        self.save_checkpoint(self.critic_two, "CriticTwo")
        self.save_checkpoint(self.target_actor, "TargetActor")
        self.save_checkpoint(self.target_critic_one, "TargetCriticOne")
        self.save_checkpoint(self.target_critic_two, "TargetCriticTwo")

    def load(self, path):
        self.load_checkpoint(self.actor_one, path, "ActorOne")
        self.load_checkpoint(self.actor_two, path, "ActorTwo")
        self.load_checkpoint(self.critic_one, path, "CriticOne")
        self.load_checkpoint(self.critic_two, path, "CriticTwo")
        self.load_checkpoint(self.target_actor, path, "TargetActor")
        self.load_checkpoint(self.target_critic_one, path, "TargetCriticOne")
        self.load_checkpoint(self.target_critic_two, path, "TargetCriticTwo")

    def name(self):
        return "td3"

    def get_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,))).to(self.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(
                self.device)
            mu = self.actor_one.forward(state).to(self.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.device)

        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def train(self, training_context):
        if training_context.cnt < self.batch_size:
            return
        else:
            self._do_train(training_context)

    def _do_train(self, training_context):
        state, action, reward, state_, done = \
            training_context.random_sample_as_tensors(
                self.batch_size, self.device)

        reward.clone().detach()
        done = T.tensor(done).to(self.device)
        state_.clone().detach()
        state.clone().detach()
        action.clone().detach()

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0],
                                 self.max_action[0])

        q1_ = self.target_critic_one.forward(state_, target_actions)
        q2_ = self.target_critic_two.forward(state_, target_actions)

        q1 = self.critic_one.forward(state, action)
        q2 = self.critic_two.forward(state, action)

        q1_[done.long()] = 0.0
        q2_[done.long()] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma * critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_one_optim.zero_grad()
        self.critic_two_optim.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_one_optim.step()
        self.critic_two_optim.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        self.actor_one_optim.zero_grad()

        actor_q1_loss = self.critic_one.forward(
            state, self.actor_one.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()

        self.actor_one_optim.step()

        agent_utils.soft_copy(
            self.critic_one, self.target_critic_one, self.tau)
        agent_utils.soft_copy(
            self.critic_two, self.target_critic_two, self.tau)
        agent_utils.soft_copy(self.actor_one, self.target_actor, self.tau)
