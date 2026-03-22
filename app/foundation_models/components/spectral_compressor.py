import torch
import torch.nn as nn

class SpectralCompressor(nn.Module):
    """
    A learnable 1x1 Convolution to reduce spectral channels which is analogous to MNF
    """

    def __init__(self, in_channels: int, compressed_channels: int):
        super().__init__()
        self.compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(compressed_channels)

    def forward(self,x):
        # X: (B, C-in, H, W)
        x = self.compress(x)
        x = self.norm(x)
        return x