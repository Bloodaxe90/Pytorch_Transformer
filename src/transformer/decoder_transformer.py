import torch
from torch import nn

from src.transformer.layer_norm import LayerNorm
from src.transformer.multi_head_attention import MultiHeadAttention
from src.transformer.position_wise_ffnn import PositionWiseFFNN


class DecoderTransformer(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads: int = 8,
        ffnn_hidden_neurons=None,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.masked_multi_head_self_attention = MultiHeadAttention(
            embedding_dim, num_heads
        )
        self.dropout1 = nn.Dropout(dropout_prob)
        self.layer_norm1 = LayerNorm(embedding_dim)

        self.position_wise_ffnn = PositionWiseFFNN(embedding_dim, ffnn_hidden_neurons)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.layer_norm2 = LayerNorm(embedding_dim)

    def forward(
        self, x: torch.Tensor, x_kv: torch.Tensor = None, mask: torch.Tensor = None
    ) -> torch.Tensor:

        identity1 = x
        attention_out = self.masked_multi_head_self_attention(x, x_kv, mask=mask)
        drop1 = self.dropout1(attention_out)
        residual1 = drop1 + identity1
        norm1 = self.layer_norm1(residual1)

        identity2 = norm1
        ffnn_out = self.position_wise_ffnn(norm1)
        drop2 = self.dropout2(ffnn_out)
        residual2 = drop2 + identity2
        norm2 = self.layer_norm2(residual2)

        return norm2
