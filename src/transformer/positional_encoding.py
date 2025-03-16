import torch
from torch import nn


class DynamicPositionEncoding(nn.Module):

    def __init__(self, max_seq_length: int = 0, embedded_dim: int = 0,
                 scaling_factor: int = 10000, device = None):
        super().__init__()
        self.scaling_factor: int = scaling_factor
        self.device = device
        self.position_encodings: torch.Tensor = self.encode((max_seq_length, embedded_dim))

    def encode(self, input_dims: tuple):
        embedded_dim: int = input_dims[-1]
        seq_length: int = input_dims[-2]

        position_encodings = torch.zeros(seq_length, embedded_dim)
        if is_odd := embedded_dim % 2 != 0: embedded_dim += 1

        raw_positions = (torch.arange(seq_length, device= self.device).unsqueeze(1) /
                         (self.scaling_factor ** (torch.arange(0, embedded_dim, 2, device= self.device) / embedded_dim)))
        position_encodings[:, 0::2] = torch.sin(raw_positions)
        position_encodings[:, 1::2] = torch.cos(raw_positions[:, :-1] if is_odd else raw_positions)
        return position_encodings

    def forward(self, x: torch.Tensor):
        if (x.shape[-2] > self.position_encodings.shape[-2] or
                x.shape[-1] != self.position_encodings.shape[-1]):
            self.position_encodings = self.encode(x.shape)
        return x + self.position_encodings[:x.shape[-2], :]


class PositionEncoding(nn.Module):

    def __init__(self, max_seq_length: int, embedded_dim: int,
                 scaling_factor: int = 10000, device = None):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.position_encodings = torch.zeros(max_seq_length, embedded_dim, device= device)

        if is_odd := embedded_dim % 2 != 0: embedded_dim += 1

        raw_positions = (torch.arange(max_seq_length, device= device).unsqueeze(1) /
                         (self.scaling_factor ** (torch.arange(0, embedded_dim, 2, device= device) / embedded_dim)))
        self.position_encodings[:, 0::2] = torch.sin(raw_positions)
        self.position_encodings[:, 1::2] = torch.cos(raw_positions[:, :-1] if is_odd else raw_positions)

    def forward(self, x: torch.Tensor):
        return x + self.position_encodings[:x.shape[-2], :]