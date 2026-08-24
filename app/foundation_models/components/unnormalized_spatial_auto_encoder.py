"""
Asanskrita - the mask-aware autoencoder that works in raw temperature units.

The only model in the roster with no PixelNormalize at all: it trains and infers
directly in Celsius. Its sibling Drashta
(normalized_masked_spatial_auto_encoder.py) is the same shape with normalisation
added, and the pair exists to test whether normalisation helps or hurts.

Why 3 input channels rather than 1: the encoder is handed
[pixels, validity_mask, input_mask]. A zeroed pixel is ambiguous on its own -
it could be off-swath, cloud-masked, or a deliberately hidden prediction target.
Comparing channels 1 and 2 lets the model tell "there is nothing here" from
"something is here and I am asking you to guess it", which is the whole basis of
masked reconstruction.

forward() returns (x_hat, z); x_hat is single-channel temperature.
Gotcha: H and W must be divisible by 2**num_stages.
"""

import torch
import torch.nn as nn
from app.foundation_models.components.spatial_encoder import SpatialEncoder
from app.foundation_models.components.spatial_decoder import SpatialDecoder


class UnNormalizedSpatialAutoencoder(nn.Module):
    """
    Unnormalized spatial autoencoder with explicit mask channels.

    Operates directly in temperature space — no pixel normalization or
    denormalization. The encoder receives a 3-channel input:

      Channel 0: thermal pixels (zeroed where input_mask == 0)
      Channel 1: validity_mask  (1=physically valid, 0=invalid)
      Channel 2: input_mask     (1=visible valid pixel, 0=hidden or invalid)

    The model learns to distinguish invalid pixels from prediction targets
    by comparing channels 1 and 2. The decoder reconstructs single-channel
    temperature output.

    Patch sizes can vary — the architecture is size-agnostic as long as
    H and W are divisible by 2^num_stages (e.g. 8 for 3 stages).

    For in_channels=1, base_channels=32, num_stages=3:

      input   (B, 3,   128, 128)   pixels + validity + input mask
        → encoder
      z       (B, 128, 16,  16)    bottleneck
        → decoder
      x_hat   (B, 1,   128, 128)   reconstructed temperature
    """

    def __init__(self, in_channels=1, base_channels=32, num_stages=3, kernel_size=4):
        super().__init__()
        # Encoder has to take in 3 channels : pixels + validity + input mask
        self.encoder = SpatialEncoder(in_channels + 2, base_channels, num_stages, kernel_size=kernel_size)
        # out_channels = in_channels so decoder reconstructs to original channel count
        self.decoder = SpatialDecoder(in_channels, base_channels, num_stages, kernel_size=kernel_size)

    def forward(self, x, validity_mask=None, input_mask=None):
        if validity_mask is not None:
            x = x * validity_mask
        if input_mask is not None:
            x = x * input_mask
        model_input = torch.cat([x, validity_mask, input_mask], dim = 1) # (B, 3, H,  W)
        z = self.encoder(model_input)       # (B, bc * 2^(num_stages-1), H/2^n, W/2^n)
        x_hat = self.decoder(z)
        return x_hat, z
