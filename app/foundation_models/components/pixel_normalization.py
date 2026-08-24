"""
Per-band z-score normalisation layers used by every model that has one.

Wrapped by SpatialAutoencoder, NormalizedMaskedSpatialAutoencoder, SegFormerMAE
and HyperspectralSegFormerMAE. Stats come from app/constants/*.json at training
time, or from an InferenceConfig.pixel_stats_override at inference time.

Why this matters more than it looks: the stats are `register_buffer`s, not plain
attributes, so they are part of the module's state_dict and travel INSIDE the
checkpoint .pt file. You cannot swap normalisation stats without rebuilding the
model - which is exactly why PixelStatsOverride exists for uncalibrated sensors
whose values sit nowhere near the baked-in distribution.

Shapes: mean/std are stored (1, C, 1, 1) so they broadcast against (B, C, H, W).
"""

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
