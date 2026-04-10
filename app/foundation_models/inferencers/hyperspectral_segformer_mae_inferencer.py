"""
Inferencer for the Hyperspectral SegFormer MAE reconstruction model.

Extends the thermal SegFormerMAEInferencer with:
    - HyperspectralSegFormerMAE model (with spectral compressor/decompressor)
    - Dual anomaly scoring: L1 residual (magnitude) + SAM residual (spectral shape)
    - Per-band pixel stats for normalisation

The two-pass masking, full-scene sliding window, overlap-averaging,
and mask erosion logic are inherited from SegFormerMAEInferencer.
"""

import json
import logging

import torch
import torch.nn as nn
import numpy as np

from app.foundation_models.inferencers.segformer_mae_inferencer import (
    SegFormerMAEInferencer,
)
from app.foundation_models.components.hyperspectral_seg_former_mae import (
    HyperspectralSegFormerMAE,
)
from app.models.training.training_config import HyperspectralSegFormerMAEConfig

logger = logging.getLogger("HyperspectralSegFormerMAEInferencer")


class HyperspectralSegFormerMAEInferencer(SegFormerMAEInferencer):
    """
    Hyperspectral SegFormer MAE inferencer.

    Inherits all masking and full-scene logic from SegFormerMAEInferencer.
    Only overrides build_model() and adds spectral anomaly scoring.
    """

    def build_model(self) -> nn.Module:
        cfg: HyperspectralSegFormerMAEConfig = self.config.model_config_
        pixel_mean, pixel_std = None, None
        stats_path = self.config.pixel_stats_path
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
            pixel_mean = stats["mean"]
            pixel_std = stats["std"]
            logger.info(
                "Per-band pixel stats loaded: %d means, %d stds",
                len(pixel_mean), len(pixel_std),
            )
        return HyperspectralSegFormerMAE(
            in_channels=cfg.in_channels,
            compressed_channels=cfg.compressed_channels,
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            reduction_ratios=cfg.reduction_ratios,
            num_blocks=cfg.num_blocks,
            decoder_dim=cfg.decoder_dim,
            expansion_ratio=cfg.expansion_ratio,
            drop_rate=0.0,  # No dropout at inference
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

    def compute_anomaly_scores(
        self,
        original: torch.Tensor,
        reconstruction: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        Compute dual anomaly scores from reconstruction residuals.

        Args:
            original:       (C, H, W) or (B, C, H, W) — input reflectance
            reconstruction: same shape — model reconstruction
            mask:           (1, H, W) or (B, 1, H, W) — validity mask

        Returns:
            dict with:
                'l1':  (H, W) or (B, H, W) — per-pixel mean absolute error across bands
                'sam': (H, W) or (B, H, W) — per-pixel spectral angle in radians
        """
        squeeze = False
        if original.dim() == 3:
            original = original.unsqueeze(0)
            reconstruction = reconstruction.unsqueeze(0)
            mask = mask.unsqueeze(0)
            squeeze = True

        # L1: mean absolute error across spectral bands
        l1 = (original - reconstruction).abs().mean(dim=1)  # (B, H, W)

        # SAM: spectral angle between original and reconstruction
        mask_bc = mask.expand_as(original)  # (B, C, H, W)
        dot = (original * reconstruction * mask_bc).sum(dim=1, keepdim=True)
        norm_o = (original * original * mask_bc).sum(dim=1, keepdim=True).sqrt()
        norm_r = (reconstruction * reconstruction * mask_bc).sum(dim=1, keepdim=True).sqrt()
        cos_sim = (dot / (norm_o * norm_r + 1e-8)).clamp(-1.0, 1.0)
        sam = torch.acos(cos_sim).squeeze(1)  # (B, H, W)

        # Zero out invalid pixels
        spatial_mask = mask[:, 0]  # (B, H, W)
        l1 = l1 * spatial_mask
        sam = sam * spatial_mask

        if squeeze:
            l1 = l1.squeeze(0)
            sam = sam.squeeze(0)

        return {"l1": l1, "sam": sam}
