"""
Concrete trainer for the SpatialAutoencoder.

Implements build_model(), compute_loss(), and validation_step()
for masked spatial reconstruction on single-band thermal patches.

Patches with less than 40% valid pixels are discarded from the batch
before training — they are mostly invalid and would add noise.
"""

import torch
import torch.nn as nn

from app.abstract_classes.foundation_trainer import FoundationTrainer
from app.foundation_models.components.spatial_auto_encoder import SpatialAutoencoder
from app.models.training.training_config import (
    TrainingConfig,
    SpatialAutoencoderConfig,
)

MIN_VALID_PIXEL_FRACTION = 0.4


class SpatialAutoencoderTrainer(FoundationTrainer):

    def build_model(self) -> nn.Module:
        cfg: SpatialAutoencoderConfig = self.config.model_config_
        return SpatialAutoencoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            num_stages=cfg.num_stages,
        )

    def _build_mask(self, batch: dict) -> torch.Tensor:
        """Combine pure validity and custom quality masks."""
        pure_validity = batch["pure_validity_mask.npy"].to(self.device)    # (B, 1, H, W)
        custom_quality = batch["custom_quality_mask.npy"].to(self.device)  # (B, 1, H, W)
        return (pure_validity * custom_quality).float()                      # (B, 1, H, W)

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
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        Masked MSE reconstruction loss with patch-level filtering.

        Patches with < 40% valid pixels are discarded. Invalid pixels
        in surviving patches are zeroed before the forward pass so they
        don't skew BatchNorm statistics. The mask excludes them from the
        loss so the network only learns from valid surface observations.

        Returns (loss, num_valid_samples) so the training loop can
        track how many samples actually contributed to training.

        Loss = sum((x_hat - x)^2 * mask) / sum(mask)
        """
        pixels = batch["pixels.npy"].to(self.device)  # (B, C, H, W)
        mask = self._build_mask(batch)                 # (B, 1, H, W)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)

        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        # Zero invalid pixels before forward pass
        x = pixels * mask
        x_hat, _ = model(x)
        
        # Note that we are computing the average loss
        # per valid pixel across batches, so we multiply by num_kept to get the total MSE loss for this batch.
        # This is done downstream 
        loss = ((x_hat - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return loss, num_kept

    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        """Same masked MSE with filtering, returns (loss_value, num_valid_samples)."""
        loss, num_kept = self.compute_loss(batch, model)
        return loss.item(), num_kept
