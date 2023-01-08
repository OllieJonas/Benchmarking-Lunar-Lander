import torch.nn as nn
import torch.nn.functional as F

class StateActionNetwork(nn.Module):
    """
     Neural Network to store state action values
    """
    def __init__(self, state_dim, action_dim):
        super(StateActionNetwork, self).__init__()
        self.num_hidden_units = 256

        self.input_layer = nn.Linear(state_dim, self.num_hidden_units)
        self.output_layer = nn.Linear(self.num_hidden_units, action_dim)

    def forward(self, state):
        q_vals = F.relu(self.input_layer(state))
        q_vals = self.output_layer(q_vals)

        return q_vals