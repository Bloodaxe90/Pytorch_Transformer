import torch
from torch import nn


class DynamicPositionEncoding(nn.Module):

    def __init__(self, scaling_factor: int = 10000):
        super().__init__()
        self.scaling_factor: int = scaling_factor

        self.position_encodings: torch.Tensor = torch.empty(0)

    def encode(self, seq_length: int, embedded_dim: int, device: str):
        if is_odd := embedded_dim % 2 != 0:
            embedded_dim += 1

        position_encodings = torch.zeros(seq_length, embedded_dim).to(device)

        raw_positions = torch.arange(seq_length).unsqueeze(1) / (
            self.scaling_factor ** (torch.arange(0, embedded_dim, 2) / embedded_dim)
        )
        position_encodings[:, 0::2] = torch.sin(raw_positions)
        position_encodings[:, 1::2] = torch.cos(
            raw_positions[:, :-1] if is_odd else raw_positions
        )
        return position_encodings

    def forward(self, x: torch.Tensor):
        seq_length, embedding_dim = x.shape[-2], x.shape[-1]

        if (
            seq_length > self.position_encodings.shape[0]
            or embedding_dim != self.position_encodings.shape[1]
        ):
            self.position_encodings = self.encode(
                seq_length, embedding_dim, x.device.type
            )
        return x + self.position_encodings[:seq_length, :]


class PositionEncoding(nn.Module):

    def __init__(
        self, max_seq_length: int, embedded_dim: int, scaling_factor: int = 10000
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.position_encodings = torch.zeros(max_seq_length, embedded_dim)

        if is_odd := embedded_dim % 2 != 0:
            embedded_dim += 1

        raw_positions = torch.arange(max_seq_length).unsqueeze(1) / (
            self.scaling_factor ** (torch.arange(0, embedded_dim, 2) / embedded_dim)
        )
        self.position_encodings[:, 0::2] = torch.sin(raw_positions)
        self.position_encodings[:, 1::2] = torch.cos(
            raw_positions[:, :-1] if is_odd else raw_positions
        )

    def forward(self, x: torch.Tensor):
        return x + self.position_encodings[: x.shape[-2], :]
