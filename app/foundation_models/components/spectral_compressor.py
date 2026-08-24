"""
The spectral bottleneck that makes hyperspectral transformers affordable.

Used only by HyperspectralSegFormerMAE (Indradhanu), which squeezes 165 bands
down to 24-32 before the encoder and expands back afterwards.

Why compress at all: attention cost scales with the token feature dimension, and
165 bands is far more than the spatial encoder needs. A learned 1x1 convolution
is a trainable analogue of MNF - it finds the few spectral combinations that
actually carry signal, but optimised end-to-end for reconstruction rather than
for signal-to-noise.

The asymmetry is deliberate and easy to "tidy" by mistake: the compressor has
BatchNorm, the decompressor has NO norm and NO activation. The compressor's job
is to hand the encoder a stable distribution; the decompressor's output feeds
denormalisation and must stay free to represent any value.
"""

import torch
import torch.nn as nn


class SpectralCompressor(nn.Module):
    """
    A learnable 1x1 Convolution to reduce spectral channels — analogous to MNF
    but trained end-to-end for the reconstruction objective.

    (B, C_in, H, W) → (B, D, H, W) where D = compressed_channels
    """

    def __init__(self, in_channels: int, compressed_channels: int):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(compressed_channels)

    def forward(self, x):
        x = self.compress(x)
        x = self.norm(x)
        return x


class SpectralDecompressor(nn.Module):
    """
    Inverse spectral projection — expands compressed channels back to
    full spectral dimension. No activation and no BatchNorm — output
    must be free to represent any value in the normalised range.

    (Asymmetric with SpectralCompressor which has BatchNorm: the compressor
    stabilises input distribution for the encoder; the decompressor output
    feeds into denormalisation which requires unconstrained values.)

    (B, D, H, W) → (B, C_out, H, W)
    """

    def __init__(self, compressed_channels: int, out_channels: int):
        super().__init__()
        self.decompress = nn.Conv2d(compressed_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.decompress(x)