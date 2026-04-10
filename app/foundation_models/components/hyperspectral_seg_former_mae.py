"""
Hyperspectral SegFormer MAE — reconstruction model for multi-band anomaly detection.

Extends the thermal SegFormerMAE with a learned spectral compression bottleneck:

    Input: (B, 165, H, W)
        → Mask + Normalise (per-band z-score)
        → SpectralCompressor: Conv2d(165, D, 1)     [learned MNF]
        → SegFormer Encoder (4-stage, in_channels=D)
        → SegFormer Decoder (PixelShuffle, out_channels=D)
        → SpectralDecompressor: Conv2d(D, 165, 1)   [inverse projection]
        → Denormalise
    Output: (B, 165, H, W)

The compressor/decompressor keep the encoder-decoder small and efficient
while the model learns an optimal spectral basis end-to-end.
"""

import torch
import torch.nn as nn

from app.foundation_models.components.seg_former_encoder import SegFormerEncoder
from app.foundation_models.components.seg_former_decoder import SegFormerDecoder
from app.foundation_models.components.spectral_compressor import (
    SpectralCompressor,
    SpectralDecompressor,
)
from app.foundation_models.components.pixel_normalization import (
    PixelNormalize,
    PixelDenormalize,
)


class HyperspectralSegFormerMAE(nn.Module):
    """
    SegFormer MAE with learned spectral compression for hyperspectral input.

    Args:
        in_channels:          Full spectral band count (e.g. 165)
        compressed_channels:  Bottleneck dimension D (e.g. 24)
        embed_dims:           Embedding dims per encoder stage
        num_heads:            Attention heads per stage
        reduction_ratios:     ESA spatial reduction per stage
        num_blocks:           Transformer blocks per stage
        decoder_dim:          Decoder fusion channel dim
        expansion_ratio:      MixFFN expansion multiplier
        drop_rate:            Dropout rate
        pixel_mean:           Per-band dataset mean (list of in_channels floats)
        pixel_std:            Per-band dataset std (list of in_channels floats)
    """

    def __init__(
        self,
        in_channels=165,
        compressed_channels=24,
        embed_dims=None,
        num_heads=None,
        reduction_ratios=None,
        num_blocks=None,
        decoder_dim=256,
        expansion_ratio=4,
        drop_rate=0.0,
        pixel_mean=None,
        pixel_std=None,
    ):
        super().__init__()

        if embed_dims is None:
            embed_dims = [32, 64, 160, 256]
        if num_heads is None:
            num_heads = [1, 2, 5, 8]
        if reduction_ratios is None:
            reduction_ratios = [8, 4, 2, 1]
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]

        self.in_channels = in_channels
        self.compressed_channels = compressed_channels

        # Per-band normalisation (registered buffers, not trainable)
        self.normalize = (
            PixelNormalize(pixel_mean, pixel_std)
            if pixel_mean is not None
            else None
        )
        self.denormalize = (
            PixelDenormalize(pixel_mean, pixel_std)
            if pixel_mean is not None
            else None
        )

        # Learned spectral compression: 165 → D
        self.compressor = SpectralCompressor(in_channels, compressed_channels)

        # Encoder operates in D-dimensional spectral space
        self.encoder = SegFormerEncoder(
            in_channels=compressed_channels,
            embed_dims=embed_dims,
            num_heads=num_heads,
            reduction_ratios=reduction_ratios,
            num_blocks=num_blocks,
            expansion_ratio=expansion_ratio,
            drop_rate=drop_rate,
        )

        # Decoder reconstructs in D-dimensional spectral space
        self.decoder = SegFormerDecoder(
            embed_dims=embed_dims,
            decoder_dim=decoder_dim,
            out_channels=compressed_channels,
        )

        # Inverse spectral projection: D → 165
        self.decompressor = SpectralDecompressor(compressed_channels, in_channels)

    def forward(self, x, mask=None, keep_mask=None):
        """
        Args:
            x:         (B, C, H, W) — input reflectance cube (C = in_channels)
            mask:      (B, 1, H, W) — pixel validity mask (1=valid, 0=invalid)
            keep_mask: (B, N) — token-level keep mask for MAE (1=keep, 0=remove)

        Returns:
            x_hat: (B, C, H, W) — reconstructed reflectance cube
        """
        # Zero out invalid pixels before normalisation
        if mask is not None:
            x = x * mask
        # x: (B, 165, H, W) — invalid pixels now 0.0

        # Per-band z-score normalisation
        if self.normalize is not None:
            x = self.normalize(x)
        # x: (B, 165, H, W) — normalised space

        # Spectral compression: 165 → D
        x = self.compressor(x)
        # x: (B, D, H, W) — compressed spectral space

        # Encode with optional MAE token removal at Stage 1
        features = self.encoder(x, keep_mask=keep_mask)

        # Decode: fuse multi-scale features, PixelShuffle to full resolution
        x_hat = self.decoder(features)
        # x_hat: (B, D, H, W) — reconstruction in compressed space

        # Spectral decompression: D → 165
        x_hat = self.decompressor(x_hat)
        # x_hat: (B, 165, H, W) — reconstruction in normalised spectral space

        # Denormalise back to physical reflectance
        if self.denormalize is not None:
            x_hat = self.denormalize(x_hat)
        # x_hat: (B, 165, H, W) — reconstruction in reflectance space

        return x_hat
