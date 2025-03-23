import torch
from torch import nn
import torch.nn.functional as F
from src.transformer.decoder_transformer import DecoderTransformer
from src.transformer.positional_encoding import DynamicPositionEncoding


class CharacterGenerator(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_transformers: int,
        num_heads: int = 8,
        ffnn_hidden_neurons: int = None,
        dropout_prob: float = 0.2,
    ):
        super().__init__()
        assert num_transformers > 0, "Must have at least one Transformer "
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = DynamicPositionEncoding()
        self.dropout = nn.Dropout(dropout_prob)

        self.decoder_transformers = nn.ModuleList(
            [
                DecoderTransformer(
                    embedding_dim, num_heads, ffnn_hidden_neurons, dropout_prob
                )
                for _ in range(num_transformers)
            ]
        )
        self.post_processing = nn.Linear(embedding_dim, vocab_size)

    def generate(self, x: torch.Tensor):
        self.eval()
        with torch.inference_mode():
            logits = self.forward(x.unsqueeze(0))
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = x.shape[:2]

        embedded_x = self.embedding(x)
        position_encoded_x = self.position_encoding(embedded_x)
        drop_x = self.dropout(position_encoded_x)

        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1) * -1e8
        causal_mask = (
            causal_mask.unsqueeze(0).expand(batch_size, -1, -1).to(x.device.type)
        )

        context_aware_x = drop_x
        for decoder_transformer in self.decoder_transformers:
            context_aware_x = decoder_transformer(context_aware_x, mask=causal_mask)

        return self.post_processing(context_aware_x)
