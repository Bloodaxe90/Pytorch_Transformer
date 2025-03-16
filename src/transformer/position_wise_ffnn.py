import torch
from torch import nn


class PositionWiseFFNN(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, embedding_dim)
        self.linear2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.linear1(x))

