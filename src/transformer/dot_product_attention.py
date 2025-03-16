import math

import torch
from torch import nn
class DotProductAttention(nn.Module):

    def __init__(self, embedded_dim: int, output_dim: int = None, scaled: bool = True):
        super().__init__()
        self.output_dim = output_dim if output_dim is not None else embedded_dim
        self.scaled = scaled

        self.q_layer = nn.Linear(embedded_dim, output_dim)
        self.k_layer = nn.Linear(embedded_dim, output_dim)
        self.v_layer = nn.Linear(embedded_dim, output_dim)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x: torch.Tensor, x_kv: torch.Tensor = None,
                mask: torch.Tensor = None,) -> torch.Tensor:
        x_kv = x if x_kv is None else x_kv #if x_kv is not none cross attention will be used

        q = self.q_layer(x)
        k = self.k_layer(x_kv)
        v = self.v_layer(x_kv)

        raw_attention = torch.matmul(q, k.transpose(-1, -2)) / (math.sqrt(
            self.output_dim) if self.scaled else 1)

        if mask is not None:
            assert mask.shape == (x.shape[0], x.shape[1], x_kv.shape[1]), "Mask does not match attention shape"
            raw_attention += mask

        attention_scores = self.softmax(raw_attention)
        return torch.matmul(attention_scores, v)







