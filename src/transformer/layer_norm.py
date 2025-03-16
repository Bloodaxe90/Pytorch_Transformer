import torch
from torch import nn


class LayerNorm(nn.Module):

    def __init__(self, embedded_dim: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(embedded_dim))
        self.beta = nn.Parameter(torch.zeros(embedded_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(x, dim= -1, keepdim=True)
        sd = torch.sqrt(torch.var(x, dim= -1, keepdim=True, unbiased=False) + self.epsilon)
        return (self.gamma * ((x - mean) / sd)) + self.beta