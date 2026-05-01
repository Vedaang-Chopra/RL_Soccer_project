import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size, hidden_layers):
        super().__init__()
        layers = []
        last_size = obs_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            last_size = hidden_size
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(last_size, action_size)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output(x)
