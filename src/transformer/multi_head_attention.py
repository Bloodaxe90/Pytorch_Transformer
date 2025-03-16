import torch
from torch import nn

from src.transformer.dot_product_attention import \
    DotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, embedded_dim: int, num_heads: int = 8, scaled: bool = True):
        super().__init__()
        assert num_heads > 0, "Number of heads must be greater than 0"
        head_dim: int = int(embedded_dim / num_heads)
        assert head_dim == embedded_dim / num_heads and head_dim > 0, \
            f"The head dimensions ({embedded_dim}/{num_heads}) must be an integer greater than 0"

        self.attention_heads = nn.ModuleList(
            [DotProductAttention(embedded_dim, head_dim, scaled)
             for _ in range(num_heads)]
        )
        self.output_layer = nn.Linear(embedded_dim, embedded_dim)

    def forward(self, x: torch.Tensor, x_kv: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        return self.output_layer(torch.cat(
            [attention_head(x, x_kv, mask) for attention_head in self.attention_heads], dim = -1
        ))



