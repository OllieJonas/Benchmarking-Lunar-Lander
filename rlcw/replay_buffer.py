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
        self.buffer_s = []
        

    def add_to_buffer_sarsa(self, data):
        #data must be of the form (state,next_state,action,n_action,reward,terminal)
        self.buffer_s.append(data)


    def sample_minibatch_sarsa(self,minibatch_length):
        
        states = []
        next_states = []
        actions = []
        next_actions = []
        rewards = []
        terminals = []
        for i in range(minibatch_length):
            
            random_int = np.random.randint(0, len(self.buffer_s)-1) 
        
            transition = self.buffer_s[random_int]
            '''
            states = np.append(states, transition[0])
            next_states = np.append(next_states, transition[1])
            actions = np.append(actions, transition[2])
            next_actions = np.append(next_actions, transition[3])
            rewards = np.append(rewards, transition[4])
            terminals = np.append(terminals, transition[5])
            '''
            states.append(transition[0])
            next_states.append(transition[1])
            actions.append(transition[2])
            next_actions.append(transition[3])
            rewards.append(transition[4])
            terminals.append(transition[5])
            
        #states = np.squeeze(states)
        #next_states = np.squeeze(next_states)
        #print("hi")
        #print(states)
        '''
        print("next states")
        print(next_states)
        #print(torch.Tensor(states))
        print("action")
        print(actions)
        print("next_actions")
        print(next_actions)
        print("reward")
        print(reward)
        print("terminal")states, next_states, actions, next_actions, reward, terminals
        print(terminal)'''
        # this corrects the array randomly being considered an object
        for i in range(0, len(states)):
            if len(states[i]) == 2:
                states[i] = states[i][0]
        '''
        try:
            torch.Tensor(states)
        except:
            #print([states[0]])
            #print([next_states[0]])
            if len(states[0]) == 2:
                states[0] = states[0][0]
            #print(len(states[0]))
            return torch.Tensor([states[0]]), torch.Tensor([next_states[0]]), torch.Tensor([actions[0]]), torch.Tensor([next_actions[0]]), torch.Tensor([rewards[0]]), torch.Tensor([terminals[0]])
         '''   
        
        return torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(next_actions), torch.Tensor(rewards), torch.Tensor(terminals)#torch.from_numpy(states), torch.from_numpy(next_states), torch.from_numpy(actions), torch.from_numpy(next_actions), torch.from_numpy(rewards), torch.from_numpy(terminals)# torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(actions), torch.Tensor(next_actions), torch.Tensor(rewards), torch.Tensor(terminals)

    def add(self, state, next_state, action, reward, done, invert_done=True):
        index = self.cnt % self.max_capacity
        self.states[index] = state
        self.next_states[index] = next_state
        self.actions[index] = action
        self.rewards[index] = reward

        self.dones[index] = 1 - done if invert_done else done
        self.cnt += 1

    def add_sarsa(self, state, next_state, action, next_action, reward, done, invert_done=True):
        index = self.cnt % self.max_capacity
        '''
        self.states = np.append(self.states, state)
        self.next_states = np.append(self.next_states, next_state)
        self.actions = np.append(self.actions, action)
        self.next_actions = np.append(self.next_actions, next_action)
        self.rewards = np.append(self.rewards, reward)
        self.dones = np.append(self.dones, done)
        print(self.states)

        
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.next_actions.append(next_action)
        self.rewards.append(reward)
        '''
        index = self.cnt % self.max_capacity
        #import pdb; pdb.set_trace();
        self.states[index] = state[0]
        self.next_states[index] = next_state[0]
        self.actions[index] = action
        self.next_actions[index] = next_action
        #import pdb; pdb.set_trace();
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

    def random_sample_sarsa(self, sample_size):
        size = min(self.cnt, self.max_capacity)
        
        batch = np.random.choice(size, sample_size)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        next_states = self.next_states[batch]
        next_action = self.next_actions[batch]
        terminal = self.dones[batch]

        new_actions = []
        new_next_actions = []
        new_rewards = []
        new_terminal = []
        #import pdb; pdb.set_trace();
    
        for i in range(0, len(actions)):
            new_actions.append([actions[i]])
            new_next_actions.append([next_action[i]])
            new_rewards.append([rewards[i]])
            new_terminal.append([terminal[i]])
            if len(states[i]) == 2:
                states[i] = states[i][0]
       
        return torch.Tensor(states), torch.Tensor(next_states), torch.Tensor(new_actions), torch.Tensor(new_next_actions), torch.Tensor(new_rewards), torch.Tensor(new_terminal)

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
