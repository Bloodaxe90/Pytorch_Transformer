import torch
from torch import nn

from src.transformer.layer_norm import LayerNorm
from src.transformer.multi_head_attention import MultiHeadAttention
from src.transformer.position_wise_ffnn import PositionWiseFFNN
from src.transformer.positional_encoding import PositionEncoding


class CharacterGenerator(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = PositionEncoding(100, embedding_dim)

        self.masked_multi_head_self_attention = MultiHeadAttention(embedding_dim, 8)
        self.layer_norm1 = LayerNorm(embedding_dim)

        self.position_wise_ffnn = PositionWiseFFNN(embedding_dim)
        self.layer_norm2 = LayerNorm(embedding_dim)

        self.linear1 = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:2]

        embedded_x = self.embedding(x)
        position_encoded_x = self.position_encoding(embedded_x)

        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1) * -1e8
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        identity1 = position_encoded_x
        residual_x1 = self.masked_multi_head_self_attention(position_encoded_x, mask= mask) + identity1
        norm_x1 = self.layer_norm1(residual_x1)

        identity2 = norm_x1
        residual_x2 = self.position_wise_ffnn(norm_x1) + identity2
        norm_x2 = self.layer_norm2(residual_x2)

        return self.linear1(norm_x2)


