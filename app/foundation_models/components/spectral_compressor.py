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