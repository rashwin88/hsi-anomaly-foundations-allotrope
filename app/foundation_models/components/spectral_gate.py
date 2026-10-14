"""
Per-pixel band gating: each pixel scales its own bands between 0 and 1 before the fixed 1x1
compressor.

Variant B of the SpectralEarth pretraining experiment (research/experiments/
indradhanu-se-pretrain/): placed in front of SpectralCompressor, it gives every pixel its own
recipe, W . diag(g(x)), at about three times the compressor's arithmetic. It exists to answer
one question cheaply - does per-pixel adaptivity alone explain any gain the cross-attention
stem (spectral_attention.py) shows? Taught in `indradhanu workings/39-per-pixel-gating.md`.

Gotchas:
  - Tied to a fixed band count, like SpectralCompressor; it knows "input 87", not 1650 nm.
  - Sigmoid never reaches 0, so it cannot skip a dead band exactly; only masking can.
  - Gates start near 0.5, halving the input; the compressor's BatchNorm absorbs the scale.
"""

import torch
import torch.nn as nn


class SpectralGate(nn.Module):
    """(B, N, H, W) -> same shape, each band of each pixel scaled by
    sigmoid(Conv1x1 -> GELU -> Conv1x1) of that pixel's spectrum. Parameters for N = 202,
    hidden = 32: 13,162."""

    def __init__(self, bands: int, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(bands, hidden, kernel_size=1), nn.GELU(), nn.Conv2d(hidden, bands, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.score(x))
