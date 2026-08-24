"""
Drashta - Asanskrita's architecture plus z-score normalisation.

Identical 3-channel mask design to unnormalized_spatial_auto_encoder.py
(see that file for why the validity and input masks are separate channels),
with PixelNormalize on the way in and PixelDenormalize on the way out.

Why normalisation is expected to help detection, not just training: mapping into
z-score space means an unusually hot pixel arrives as a large z-value, far from
anything the model saw while learning ordinary terrain. That makes it HARDER to
reconstruct from spatial context - which is precisely what we want, since the
reconstruction error is the anomaly score.

The normalisation stats live in the checkpoint as registered buffers, so this
model cannot be pointed at a differently-calibrated sensor without either
retraining or an InferenceConfig.pixel_stats_override.

forward() returns (x_hat, z), x_hat denormalised back to temperature.
"""

import torch
import torch.nn as nn
from app.foundation_models.components.spatial_encoder import SpatialEncoder
from app.foundation_models.components.spatial_decoder import SpatialDecoder
from app.foundation_models.components.pixel_normalization import PixelDenormalize, PixelNormalize


class NormalizedMaskedSpatialAutoencoder(nn.Module):
    """
    Normalized spatial autoencoder with explicit mask channels.

    Combines pixel normalization (z-score) with the 3-channel mask
    architecture. The encoder receives:

      Channel 0: normalized thermal pixels (zeroed where input_mask == 0)
      Channel 1: validity_mask  (1=physically valid, 0=invalid)
      Channel 2: input_mask     (1=visible valid pixel, 0=hidden or invalid)

    Normalization maps temperatures into z-score space before encoding,
    making anomalous temperatures (high z-scores) harder to reconstruct
    from spatial context. The decoder output is denormalized back to
    temperature space.

    For in_channels=1, base_channels=32, num_stages=3:

      input   (B, 3,   128, 128)   normalized pixels + validity + input mask
        -> encoder
      z       (B, 128, 16,  16)    bottleneck
        -> decoder
      x_hat   (B, 1,   128, 128)   reconstructed temperature (denormalized)
    """

    def __init__(self, in_channels=1, base_channels=32, num_stages=3,
                 pixel_mean=None, pixel_std=None, kernel_size=4):
        super().__init__()
        self.normalize = PixelNormalize(pixel_mean, pixel_std) if pixel_mean is not None else None
        self.denormalize = PixelDenormalize(pixel_mean, pixel_std) if pixel_mean is not None else None
        # Encoder takes 3 channels: normalized pixels + validity + input mask
        self.encoder = SpatialEncoder(in_channels + 2, base_channels, num_stages, kernel_size=kernel_size)
        self.decoder = SpatialDecoder(in_channels, base_channels, num_stages, kernel_size=kernel_size)

    def forward(self, x, validity_mask=None, input_mask=None):
        if validity_mask is not None:
            x = x * validity_mask
        if input_mask is not None:
            x = x * input_mask
        if self.normalize is not None:
            x = self.normalize(x)
        model_input = torch.cat([x, validity_mask, input_mask], dim=1)  # (B, 3, H, W)
        z = self.encoder(model_input)
        x_hat = self.decoder(z)
        if self.denormalize is not None:
            x_hat = self.denormalize(x_hat)
        return x_hat, z
