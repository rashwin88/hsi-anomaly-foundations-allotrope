"""
Concrete trainer for the UnNormalizedSpatialAutoencoder with masked L1 loss.

Implements build_model(), compute_loss(), and validation_step()
for masked spatial reconstruction on single-band thermal patches.
Uses L1 (absolute error) loss with explicit 3-channel mask input
(pixels + validity_mask + input_mask) in raw temperature space.

Patches with less than 40% valid pixels are discarded from the batch
before training — they are mostly invalid and would add noise.
"""

import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_trainer import FoundationTrainer

logger = logging.getLogger("SpatialAutoencoderTrainer")
from app.foundation_models.components.unnormalized_spatial_auto_encoder import UnNormalizedSpatialAutoencoder
from app.models.training.training_config import (
    SpatialMaskedAutoEncoderConfig,
)

MIN_VALID_PIXEL_FRACTION = 0.4


class UnNormalizedSpatialMaskedAutoencoderTrainerL1Loss(FoundationTrainer):

    def build_model(self) -> nn.Module:
        cfg: SpatialMaskedAutoEncoderConfig = self.config.model_config_
        return UnNormalizedSpatialAutoencoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            num_stages=cfg.num_stages,
            kernel_size=cfg.kernel_size,
        )

    def _build_mask(self, batch: dict) -> torch.Tensor:
        """Combine pure validity and custom quality masks."""
        pure_validity = batch["pure_validity_mask.npy"].to(self.device)    # (B, 1, H, W)
        custom_quality = batch["custom_quality_mask.npy"].to(self.device)  # (B, 1, H, W)
        # Also picking up the modelled cloud mask
        modelled_cloud_mask = batch["predicted_cloud_mask.npy"].to(self.device)
        return (pure_validity * modelled_cloud_mask).float()                      # (B, 1, H, W)

    def _filter_batch(
        self, pixels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Discard patches with < 40% valid pixels.

        Returns filtered pixels, filtered mask, and the count of
        valid samples kept (for accurate epoch sample counting).
        """
        # mask shape: (B, 1, H, W)
        # flatten(1) collapses dims 1..3 → (B, 1*H*W) i.e. one row per sample
        # mean(dim=1) averages across that row → (B,) fraction of valid pixels per sample
        valid_fractions = mask.flatten(1).float().mean(dim=1)  # (B,)
        keep = valid_fractions >= MIN_VALID_PIXEL_FRACTION  # (B,) boolean
        num_kept = keep.sum().item()

        if num_kept == 0:
            return pixels[:0], mask[:0], 0  # empty batch

        return pixels[keep], mask[keep], num_kept

    def compute_loss(
        self, batch: dict, model: nn.Module, min_masking: float = 0.35, max_masking: float = 0.55
    ) -> tuple[torch.Tensor, int]:
        """
        Masked L1 reconstruction loss with patch-level filtering.

        Patches with < 40% valid pixels are discarded. A random prediction
        mask is generated from valid pixels, and the model receives
        validity_mask and input_mask as explicit channels.

        Returns (loss, num_valid_samples) so the training loop can
        track how many samples actually contributed to training.

        Loss = sum(|x_hat - x| * prediction_mask) / sum(prediction_mask)
        """
        pixels = batch["pixels.npy"].to(self.device)  # (B, C, H, W)
        mask = self._build_mask(batch)                 # (B, 1, H, W)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        # Get the input mask
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0
        # Generate prediction mask from valid pixels
        B = pixels.shape[0]
        mask_ratio = torch.empty(B,1,1,1, device = self.device).uniform_(min_masking, max_masking)
        rand_map = torch.rand_like(mask)
        prediction_mask = ((rand_map < mask_ratio) & (mask == 1)).float()

        input_mask = mask - prediction_mask

        # Zero invalid pixels before forward pass
        x_hat, _ = model(pixels, validity_mask =mask, input_mask=input_mask)
        
        # Note that we are computing the average loss
        # per valid pixel across batches, so we multiply by num_kept to get the total MSE loss for this batch.
        # This is done downstream 
        loss = ((x_hat - pixels).abs() * prediction_mask).sum() / prediction_mask.sum().clamp(min=1)
        return loss, num_kept
    
    def compute_validation_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        L1 reconstruction loss on all valid pixels (no random masking).

        Returns (loss, num_valid_samples).

        Loss = sum(|x_hat - x| * mask) / sum(mask)
        """
        pixels = batch["pixels.npy"].to(self.device)  # (B, C, H, W)
        mask = self._build_mask(batch)                 # (B, 1, H, W)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        # Get the input mask

        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        x_hat, _ = model(pixels, validity_mask=mask, input_mask = mask)
        
        # Note that we are computing the average loss
        # per valid pixel across batches, so we multiply by num_kept to get the total MSE loss for this batch.
        # This is done downstream 
        loss = ((x_hat - pixels).abs() * mask).sum() / mask.sum().clamp(min=1)
        return loss, num_kept

    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        """Same masked MSE with filtering, returns (loss_value, num_valid_samples)."""
        loss, num_kept = self.compute_validation_loss(batch, model)
        return loss.item(), num_kept
