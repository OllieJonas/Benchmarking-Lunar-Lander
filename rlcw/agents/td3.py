# code **inspired** by https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/TD3/td3_torch.py

import numpy as np
from typing import List

import torch as T
import torch.nn.functional as F

from td3_classes.criticNetwork import CriticNetwork
from td3_classes.actorNetwork import ActorNetwork
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
        self.chkpt_dir = config['chkpt_dir']
        self.agent_suffix = config['agent_suffix']

        self.update_actor_interval = 2
        self.warmup = 1000
        self.max_action = self.action_space.high
        self.min_action = self.action_space.low
        self.learn_step_counter = 0
        self.time_step = 0

        self.memory = ReplayBuffer(self.max_size)

        self.actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                  self.layer2_size, n_actions=self.n_actions,
                                  name='Actor', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        self.critic_one = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                        self.layer2_size, n_actions=self.n_actions,
                                        name='Critic_One', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        self.critic_two = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                        self.layer2_size, n_actions=self.n_actions,
                                        name='Critic_Two', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        self.target_actor = ActorNetwork(self.alpha, self.input_dims, self.layer1_size,
                                         self.layer2_size, n_actions=self.n_actions,
                                         name='TargetActor', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        self.target_critic_one = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                               self.layer2_size, n_actions=self.n_actions,
                                               name='TargetCritic_One', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        self.target_critic_two = CriticNetwork(self.beta, self.input_dims, self.layer1_size,
                                               self.layer2_size, n_actions=self.n_actions,
                                               name='TargetCritic_Two', chkpt_dir=self.chkpt_dir, agent_suffix=self.agent_suffix)

        #self.noise = ActionNoise(mu=np.zeros(self.n_actions))
        self.noise = config["noise"]

        self.update_network_parameters(tau=1)

    def save(self):
        pass

    def load(self):
        pass

    def name(self):
        return "td3"

    def store_memory(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def get_action(self, observation):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale=self.noise,
                                           size=(self.n_actions,)))
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)
        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise),
                                 dtype=T.float).to(self.actor.device)

        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def train(self):
        if self.memory.mem_center < self.batch_size:
            return
        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        reward = T.tensor(reward, dtype=T.float).to(self.critic_one.device)
        done = T.tensor(done).to(self.critic_one.device)
        state_ = T.tensor(new_state, dtype=T.float).to(self.critic_one.device)
        state = T.tensor(state, dtype=T.float).to(self.critic_one.device)
        action = T.tensor(action, dtype=T.float).to(self.critic_one.device)

        target_actions = self.target_actor.forward(state_)
        target_actions = target_actions + \
            T.clamp(T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0],
                                 self.max_action[0])

        q1_ = self.target_critic_one.forward(state_, target_actions)
        q2_ = self.target_critic_two.forward(state_, target_actions)

        q1 = self.critic_one.forward(state, action)
        q2 = self.critic_two.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = T.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_one.optimiser.zero_grad()
        self.critic_two.optimiser.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_one.optimiser.step()
        self.critic_two.optimiser.step()

        self.learn_step_counter += 1

        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        self.actor.optimiser.zero_grad()
        actor_q1_loss = self.critic_one.forward(
            state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimiser.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_one.named_parameters()
        critic_2_params = self.critic_two.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_one.named_parameters()
        target_critic_2_params = self.target_critic_two.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                (1-tau)*target_actor[name].clone()

        self.target_critic_one.load_state_dict(critic_1)
        self.target_critic_two.load_state_dict(critic_2)
        self.target_actor.load_state_dict(actor)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_one.save_checkpoint()
        self.critic_two.save_checkpoint()
        self.target_critic_one.save_checkpoint()
        self.target_critic_two.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_one.load_checkpoint()
        self.critic_two.load_checkpoint()
        self.target_critic_one.load_checkpoint()
        self.target_critic_two.load_checkpoint()
