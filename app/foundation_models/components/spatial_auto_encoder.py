"""
The baseline reconstruction model - encoder + decoder, single-channel input.

Ships under THREE codenames, all the same class with different training regimes:
Pratibimba (plain), Antardhana (masked, L2) and Tirohita (masked, L1). The
backend routes all three through SpatialAutoencoderInferencer.

How it detects anomalies: it is trained only to rebuild ordinary imagery, so
whatever it rebuilds badly is unlike anything it learned. The per-pixel
reconstruction error IS the anomaly score - there are no anomaly labels anywhere
in training. Scoring itself lives in app/utils/anomaly_detection/scoring.py.

forward() returns (x_hat, z). z, the bottleneck, is returned for inspection and
is not used by the scoring path.

Gotcha: H and W must be divisible by 2**num_stages. Normalisation is optional -
pass pixel_mean/pixel_std to enable it, omit them to work in raw units.
"""

import torch
import torch.nn as nn
from app.foundation_models.components.spatial_encoder import SpatialEncoder
from app.foundation_models.components.spatial_decoder import SpatialDecoder
from app.foundation_models.components.pixel_normalization import PixelDenormalize, PixelNormalize


class SpatialAutoencoder(nn.Module):
    """
    Full spatial autoencoder that wires encoder and decoder together.

    The encoder compresses spatial dimensions into a bottleneck, and the
    decoder reconstructs back to the original resolution. The forward pass
    returns both the reconstruction (x_hat) and the bottleneck (z) — z is
    useful for inspection, visualization, or downstream tasks.

    Patch sizes can vary — the architecture is size-agnostic as long as
    H and W are divisible by 2^num_stages (e.g. 8 for 3 stages).

    For in_channels=1, base_channels=32, num_stages=3:

      x       (B, 1,   128, 128)   input thermal patch
        → encoder
      z       (B, 128, 16,  16)    bottleneck: compressed spatial repr
        → decoder
      x_hat   (B, 1,   128, 128)   reconstruction

    For anomaly detection, the per-pixel anomaly score is:
      A(i,j) = (x_hat(i,j) - x(i,j))^2
    Regions the network struggles to reconstruct are flagged as anomalous.

    The encoder and decoder share the same base_channels and num_stages
    so their channel progressions are symmetric mirrors of each other.
    in_channels is passed to both so the decoder knows what to reconstruct.
    """

    def __init__(self, in_channels=1, base_channels=32, num_stages=3, pixel_mean=None, pixel_std=None, kernel_size=4):
        super().__init__()
        self.normalize = PixelNormalize(pixel_mean, pixel_std) if pixel_mean is not None else None
        self.denormalize = PixelDenormalize(pixel_mean, pixel_std) if pixel_mean is not None else None
        self.encoder = SpatialEncoder(in_channels, base_channels, num_stages, kernel_size=kernel_size)
        # out_channels = in_channels so decoder reconstructs to original channel count
        self.decoder = SpatialDecoder(in_channels, base_channels, num_stages, kernel_size=kernel_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        if self.normalize is not None:
            x = self.normalize(x)
        z = self.encoder(x)       # (B, bc * 2^(num_stages-1), H/2^n, W/2^n)
        x_hat = self.decoder(z)
        if self.denormalize is not None:
            x_hat = self.denormalize(x_hat)   # (B, in_channels, H, W)
        return x_hat, z
