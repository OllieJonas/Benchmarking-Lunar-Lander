import torch.nn as nn


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, no_actions, hidden_1_dims=256, hidden_2_dims=256):
        super(DeepQNetwork, self).__init__()
        self.input_size = input_size
        self.no_actions = no_actions

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_1_dims),
            nn.ReLU(),
            nn.Linear(hidden_1_dims, hidden_2_dims),
            nn.ReLU(),
            nn.Linear(hidden_2_dims, no_actions)
        )

    def forward(self, state):
        return self.layers.forward(state)
