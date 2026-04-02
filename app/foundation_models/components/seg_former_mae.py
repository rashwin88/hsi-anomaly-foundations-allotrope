"""
SegFormer MAE -- full reconstruction model for thermal anomaly detection.

Wires together the SegFormer encoder and MLP decoder with optional pixel
normalization/denormalization. The model takes a single-band thermal image
and reconstructs it via multi-scale transformer feature extraction.

Data flow:
    Input: (B, 1, H, W)  -- raw thermal temperatures

    1. Validity masking:  x = x * mask              zero out invalid pixels
    2. Normalize:         x = (x - mean) / std      map to z-score space
    3. Encode:            [F1, F2, F3, F4]           4-stage hierarchical features
    4. Decode:            x_hat = decode(features)   fuse and upsample to full res
    5. Denormalize:       x_hat = x_hat * std + mean back to temperature space

    Output: (B, 1, H, W)  -- reconstructed thermal temperatures

Masking strategy (random for training, checkerboard for inference) is NOT
handled inside this model. The trainer and inferencer apply masking externally
before passing the input to forward(). This keeps the model architecture clean
and reusable across different masking strategies.

For anomaly detection, the per-pixel anomaly score is:
    A(i,j) = (x(i,j) - x_hat(i,j))^2
Regions the network cannot reconstruct from spatial context produce high
reconstruction error and are flagged as anomalous.
"""

import torch
import torch.nn as nn

from app.foundation_models.components.seg_former_encoder import SegFormerEncoder
from app.foundation_models.components.seg_former_decoder import SegFormerDecoder
from app.foundation_models.components.pixel_normalization import PixelNormalize, PixelDenormalize


class SegFormerMAE(nn.Module):
    """
    SegFormer-based Masked Autoencoder for thermal reconstruction.

    Combines:
        - Optional pixel normalization (z-score via registered buffers)
        - 4-stage hierarchical transformer encoder
        - MLP reconstruction decoder with multi-scale fusion

    The model is architecture-agnostic to masking -- it simply reconstructs
    whatever input it receives. Training and inference masking strategies
    are handled by the trainer and inferencer respectively.

    Args:
        in_channels:      Input channels (1 for single-band thermal)
        embed_dims:       Embedding dimension per encoder stage, e.g. [32, 64, 160, 256]
        num_heads:        Attention heads per stage, e.g. [1, 2, 5, 8]
        reduction_ratios: ESA spatial reduction per stage, e.g. [8, 4, 2, 1]
        num_blocks:       Transformer blocks per stage, e.g. [2, 2, 2, 2]
        decoder_dim:      Common channel dim for decoder fusion (default 256)
        expansion_ratio:  MixFFN expansion multiplier (default 4)
        drop_rate:        Dropout rate for transformer blocks (default 0.0)
        pixel_mean:       Dataset mean for normalization (None to skip)
        pixel_std:        Dataset std for normalization (None to skip)
    """

    def __init__(self, in_channels=1, embed_dims=None, num_heads=None,
                 reduction_ratios=None, num_blocks=None, decoder_dim=256,
                 expansion_ratio=4, drop_rate=0.0, pixel_mean=None, pixel_std=None):
        super().__init__()

        # Defaults: SegFormer-B0 configuration
        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]
        if num_heads is None:
            num_heads = [1, 2, 5, 8]
        if reduction_ratios is None:
            reduction_ratios = [8, 4, 2, 1]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        # --- Pixel normalization ---
        # Registered buffers: saved in state_dict, moved with .to(device), not trainable
        # Maps raw temperatures to z-score space where the model operates
        self.normalize = PixelNormalize(pixel_mean, pixel_std) if pixel_mean is not None else None
        self.denormalize = PixelDenormalize(pixel_mean, pixel_std) if pixel_mean is not None else None

        # --- Encoder ---
        # 4-stage hierarchical transformer, returns [F1, F2, F3, F4]
        self.encoder = SegFormerEncoder(
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            reduction_ratios=reduction_ratios,
            num_blocks=num_blocks,
            expansion_ratio=expansion_ratio,
            drop_rate=drop_rate,
        )

        # --- Decoder ---
        # Fuses multi-scale features and upsamples to full resolution
        self.decoder = SegFormerDecoder(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            out_channels=in_channels,
        )

    def forward(self, x, mask=None, keep_mask=None):
        """
        Args:
            x:         (B, C, H, W) -- input thermal image
            mask:      (B, 1, H, W) -- optional pixel validity mask (1=valid, 0=invalid)
                       If provided, invalid pixels are zeroed before normalization.
                       Masked zeros become ~ -mean/std in normalized space, providing
                       a clear out-of-distribution signal to the encoder.
            keep_mask: (B, N) -- optional token-level keep mask for MAE (1=keep, 0=remove)
                       N = (H/4) * (W/4) tokens at Stage 1.
                       If provided, prediction target tokens are removed at Stage 1.
                       Generated by TokenMasking.generate_prediction_mask().
                       If None, no token removal (full reconstruction).

        Returns:
            x_hat: (B, C, H, W) -- reconstructed thermal image in original space
        """
        # x: (B, C, H, W)

        # Zero out invalid pixels before normalization
        # 0°C -> ~ -2.3 in normalized space (clearly out-of-distribution)
        if mask is not None:
            x = x * mask
        # x: (B, C, H, W) -- invalid pixels now 0

        # Normalize to z-score space
        if self.normalize is not None:
            x = self.normalize(x)
        # x: (B, C, H, W) -- values in approximately [-2, +2] for valid pixels

        # Encode: extract multi-scale features (with optional MAE token removal)
        features = self.encoder(x, keep_mask=keep_mask)
        # features: [F1, F2, F3, F4]
        #   F1: (B, C1, H/4,  W/4)   -- has zeros at prediction positions if keep_mask provided
        #   F2: (B, C2, H/8,  W/8)
        #   F3: (B, C3, H/16, W/16)
        #   F4: (B, C4, H/32, W/32)

        # Decode: fuse features and reconstruct
        x_hat = self.decoder(features)
        # x_hat: (B, C, H, W) -- reconstruction in normalized space

        # Denormalize back to temperature space
        if self.denormalize is not None:
            x_hat = self.denormalize(x_hat)
        # x_hat: (B, C, H, W) -- reconstruction in degrees Celsius

        return x_hat
