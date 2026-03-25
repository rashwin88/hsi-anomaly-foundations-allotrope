import torch
import torch.nn as nn

class PixelNormalize(nn.Module):
    """Normalizes input: (x - mean) / std. Stats stored as registered buffers."""

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        # (1, C, 1, 1) for broadcasting against (B, C, H, W)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class PixelDenormalize(nn.Module):
    """Inverts normalization: x * std + mean."""

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.std + self.mean
