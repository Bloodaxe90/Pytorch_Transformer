import torch
from torch import nn


class PositionWiseFFNN(nn.Module):

    def __init__(self, embedding_dim: int, hidden_neurons: int = None):
        super().__init__()
        hidden_neurons = embedding_dim if hidden_neurons is None else hidden_neurons
        self.linear1 = nn.Linear(embedding_dim, hidden_neurons)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_neurons, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))
