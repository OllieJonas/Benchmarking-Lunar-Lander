import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, input_dims=None, is_continuous=False):
        if input_dims is None:
            input_dims = [8]

        self.max_capacity = max_size
        self.input_dims = input_dims

        # buff
        self.states = np.zeros((self.max_capacity, *self.input_dims))
        self.next_states = np.zeros((self.max_capacity, *self.input_dims))
        self.actions = np.zeros((self.max_capacity, 2)) if is_continuous else np.zeros(self.max_capacity)
        self.next_actions = np.zeros((self.max_capacity, 2)) if is_continuous else np.zeros(self.max_capacity)
        self.rewards = np.zeros(self.max_capacity)
        self.dones = np.zeros(self.max_capacity, dtype=np.float32)

        self.cnt = 0
        

    def add_to_sarsa(self, state, next_state, action, next_action, reward, done, invert_done=True):
        index = self.cnt % self.max_capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.next_actions[index] = next_action
        self.rewards[index] = reward

        self.dones[index] = done
        self.cnt += 1


    def random_sample_sarsa(self, sample_size):
        
        size = min(self.cnt, self.max_capacity)
        
        batch = np.random.choice(size, sample_size)
        states, next_states, actions, next_actions, rewards, terminals = [], [], [], [], [], []

        states = self.states[batch]
        old_actions = self.actions[batch]
        old_rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        old_next_actions = self.next_actions[batch]
        old_terminals = self.dones[batch]

        # this corrects the array randomly being considered an object
        for i in range(0, len(states)):
            actions.append([old_actions[i]])
            rewards.append([old_rewards[i]])
            next_actions.append([old_next_actions[i]])
            terminals.append([old_terminals[i]])
            if len(states[i]) == 2:
                states[i] = states[i][0]

        return states, next_states, actions, next_actions, rewards, terminals

    def add(self, state, next_state, action, reward, done, invert_done=True):
        index = self.cnt % self.max_capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward

        self.dones[index] = 1 - done if invert_done else done
        self.cnt += 1
        

    def random_sample(self, sample_size):
        size = min(self.cnt, self.max_capacity)
        
        batch = np.random.choice(size, sample_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        terminal = self.dones[batch]

        return states, actions, rewards, next_states, terminal

    def random_sample_as_tensors(self, sample_size, device):
        state, action, reward, new_state, terminal = self.random_sample(sample_size)

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).to(device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(device)
        done = torch.tensor(terminal).to(device)

        return state, action, reward, new_state, terminal

    def __getitem__(self, item):
        return np.asarray([self.states[item],
                           self.next_states[item],
                           self.actions[item],
                           self.rewards[item],
                           self.dones[item]],
                          dtype=object)

    def __repr__(self):
        return f"states: {self.states.__repr__()},\n " \
               f"next_states: {self.next_states.__repr__()},\n" \
               f"actions: {self.actions.__repr__()},\n" \
               f"rewards: {self.rewards.__repr__()},\n" \
               f"dones: {self.dones.__repr__()},\n"
