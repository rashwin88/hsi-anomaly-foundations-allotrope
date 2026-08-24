"""
Trainer for the NormalizedMaskedSpatialAutoencoder.

Combines pixel z-score normalization with explicit mask channels
and L1 reconstruction loss. The model receives 3 channels
(normalized pixels + validity_mask + input_mask) and learns to
reconstruct hidden pixels from spatial context in normalized space.

Patches with less than 40% valid pixels are discarded from the batch.
"""

import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_trainer import FoundationTrainer
from app.foundation_models.components.normalized_masked_spatial_auto_encoder import (
    NormalizedMaskedSpatialAutoencoder,
)
from app.models.training.training_config import NormalizedMaskedAutoEncoderConfig

from app.foundation_models.pixel_stats import resolve_pixel_stats

logger = logging.getLogger("NormalizedMaskedAutoencoderTrainer")

MIN_VALID_PIXEL_FRACTION = 0.4


class NormalizedMaskedAutoencoderTrainer(FoundationTrainer):

    def build_model(self) -> nn.Module:
        cfg: NormalizedMaskedAutoEncoderConfig = self.config.model_config_
        pixel_mean, pixel_std = resolve_pixel_stats(self.config.data.pixel_stats_path)
        return NormalizedMaskedSpatialAutoencoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            num_stages=cfg.num_stages,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            kernel_size=cfg.kernel_size,
        )

    def _build_mask(self, batch: dict) -> torch.Tensor:
        """Combine pure validity and modelled cloud masks."""
        pure_validity = batch["pure_validity_mask.npy"].to(self.device)
        modelled_cloud_mask = batch["predicted_cloud_mask.npy"].to(self.device)
        return (pure_validity * modelled_cloud_mask).float()

    def _filter_batch(
        self, pixels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Discard patches with < 40% valid pixels."""
        valid_fractions = mask.flatten(1).float().mean(dim=1)
        keep = valid_fractions >= MIN_VALID_PIXEL_FRACTION
        num_kept = keep.sum().item()

        if num_kept == 0:
            return pixels[:0], mask[:0], 0

        return pixels[keep], mask[keep], num_kept

    def compute_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        Masked L1 reconstruction loss.

        Generates a random prediction mask from valid pixels,
        passes input_mask (visible pixels) and validity_mask to the model,
        and computes L1 loss only on the prediction targets.
        """
        cfg: NormalizedMaskedAutoEncoderConfig = self.config.model_config_
        min_masking, max_masking = cfg.masking_range

        pixels = batch["pixels.npy"].to(self.device)
        mask = self._build_mask(batch)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        B = pixels.shape[0]
        mask_ratio = torch.empty(B, 1, 1, 1, device=self.device).uniform_(min_masking, max_masking)
        rand_map = torch.rand_like(mask)
        prediction_mask = ((rand_map < mask_ratio) & (mask == 1)).float()

        input_mask = mask - prediction_mask

        x_hat, _ = model(pixels, validity_mask=mask, input_mask=input_mask)

        loss = ((x_hat - pixels).abs() * prediction_mask).sum() / prediction_mask.sum().clamp(min=1)
        return loss, num_kept

    def compute_validation_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """L1 loss on all valid pixels (no random masking)."""
        pixels = batch["pixels.npy"].to(self.device)
        mask = self._build_mask(batch)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        x_hat, _ = model(pixels, validity_mask=mask, input_mask=mask)

        loss = ((x_hat - pixels).abs() * mask).sum() / mask.sum().clamp(min=1)
        return loss, num_kept

    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        loss, num_kept = self.compute_validation_loss(batch, model)
        return loss.item(), num_kept
