"""
Trainer for the Hyperspectral SegFormer MAE reconstruction model.

Extends the thermal SegFormerMAETrainer with:
    - Spectral compression/decompression (learned MNF)
    - Combined L1 + SAM loss with configurable ramp
    - Per-band pixel normalisation
    - Hyperspectral shard format (pixels.npy, validity_cube.npy, wavelengths.npy)

The masking, token removal, and validation strategies are identical
to the thermal trainer — reused via the same TokenMasking utilities.
"""

import json
import logging
import warnings

import torch
import torch.nn as nn

# Suppress webdataset duplicate key warnings from resampled final shards
warnings.filterwarnings("ignore", message=".*duplicate file name in tar file.*")

from app.abstract_classes.foundation_trainer import FoundationTrainer
from app.foundation_models.components.hyperspectral_seg_former_mae import (
    HyperspectralSegFormerMAE,
)
from app.foundation_models.components.token_masking import TokenMasking
from app.foundation_models.components.sam_loss import SAMLoss

logger = logging.getLogger("HyperspectralSegFormerMAETrainer")

MIN_VALID_PIXEL_FRACTION = 0.4

STAGE1_KERNEL_SIZE = 4
STAGE1_STRIDE = 4
STAGE1_PADDING = 0


class HyperspectralSegFormerMAETrainer(FoundationTrainer):

    def __init__(self, config):
        super().__init__(config)
        self._current_epoch = 0
        # Accumulators for per-epoch component logging
        self._epoch_l1_sum = 0.0
        self._epoch_sam_sum = 0.0
        self._epoch_loss_samples = 0

    def _run_epoch(self, epoch, train_loaders, test_loaders):
        """Track current epoch for SAM ramp, log components, then delegate to parent."""
        self._current_epoch = epoch
        self._epoch_l1_sum = 0.0
        self._epoch_sam_sum = 0.0
        self._epoch_loss_samples = 0

        super()._run_epoch(epoch, train_loaders, test_loaders)

        # Log component losses and λ after the base class logs the combined loss
        sam_weight = self._get_sam_weight(epoch)
        if self._epoch_loss_samples > 0:
            avg_l1 = self._epoch_l1_sum / self._epoch_loss_samples
            avg_sam = self._epoch_sam_sum / self._epoch_loss_samples
            logger.info(
                f"  L1: {avg_l1:.6f} | SAM: {avg_sam:.6f} rad "
                f"({avg_sam * 180 / 3.14159:.2f} deg) | lambda: {sam_weight:.3f}"
            )

        if self._wandb_enabled:
            import wandb
            wandb.log({
                "train/l1_loss": avg_l1 if self._epoch_loss_samples > 0 else 0.0,
                "train/sam_loss": avg_sam if self._epoch_loss_samples > 0 else 0.0,
                "train/sam_loss_deg": (avg_sam * 180 / 3.14159) if self._epoch_loss_samples > 0 else 0.0,
                "train/sam_weight": sam_weight,
            }, step=epoch)

    def build_model(self) -> nn.Module:
        cfg = self.config.model_config_
        pixel_mean, pixel_std = None, None
        stats_path = self.config.data.pixel_stats_path
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
            pixel_mean = stats["mean"]  # list of 165 floats
            pixel_std = stats["std"]    # list of 165 floats
            logger.info(
                "Per-band pixel stats loaded: %d means, %d stds",
                len(pixel_mean), len(pixel_std),
            )
        model = HyperspectralSegFormerMAE(
            in_channels=cfg.in_channels,
            compressed_channels=cfg.compressed_channels,
            embed_dims=cfg.embed_dims,
            num_heads=cfg.num_heads,
            reduction_ratios=cfg.reduction_ratios,
            num_blocks=cfg.num_blocks,
            decoder_dim=cfg.decoder_dim,
            expansion_ratio=cfg.expansion_ratio,
            drop_rate=cfg.drop_rate,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        # Store SAM loss module on the trainer (not inside the model)
        self._sam_loss = SAMLoss()
        return model

    def _run_train_pass(
        self, loader, sample_cap: int
    ) -> tuple[float, int]:
        """
        Train with gradient accumulation.

        Accumulates gradients over N mini-batches before updating weights.
        Effective batch size = batch_size × gradient_accumulation_steps.
        Loss is scaled by 1/N so the gradient magnitude matches a single
        large batch, keeping the learning rate meaningful.
        """
        accum_steps = self.config.data.gradient_accumulation_steps
        if accum_steps <= 1:
            return super()._run_train_pass(loader, sample_cap)

        total_loss = 0.0
        valid_samples = 0
        accum_count = 0

        self.optimizer.zero_grad()

        for batch in loader:
            if valid_samples >= sample_cap:
                break

            loss, num_kept = self.compute_loss(batch, self.model)

            if num_kept == 0:
                continue

            # Scale loss by accumulation steps so gradient magnitude
            # matches what a single large batch would produce
            scaled_loss = loss / accum_steps
            scaled_loss.backward()

            total_loss += loss.item() * num_kept
            valid_samples += num_kept
            accum_count += 1

            if accum_count % accum_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Flush any remaining accumulated gradients
        if accum_count % accum_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return total_loss, valid_samples

    def _build_mask(self, batch: dict) -> torch.Tensor:
        """
        Build spatial validity mask from the validity cube.

        Hyperspectral validity is pixel-level binary (all 165 bands agree
        after the preprocessing pipeline). Band 0 is a sufficient proxy.
        Returns (B, 1, H, W) float mask.
        """
        validity = batch["validity_cube.npy"].to(self.device)
        # Use band 0 as spatial validity proxy: (B, C, H, W) → (B, 1, H, W)
        return validity[:, 0:1, :, :].float()

    def _filter_batch(
        self, pixels: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        valid_fractions = mask.flatten(1).float().mean(dim=1)
        keep = valid_fractions >= MIN_VALID_PIXEL_FRACTION
        num_kept = keep.sum().item()
        if num_kept == 0:
            return pixels[:0], mask[:0], 0
        return pixels[keep], mask[keep], num_kept

    def _build_prediction_mask(
        self, pixel_mask: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_mask = TokenMasking.pixel_mask_to_token_mask(
            pixel_mask,
            kernel_size=STAGE1_KERNEL_SIZE,
            stride=STAGE1_STRIDE,
            padding=STAGE1_PADDING,
        )
        keep_mask, pred_mask = TokenMasking.generate_prediction_mask(
            token_mask, mask_ratio, self.device
        )
        return keep_mask, pred_mask

    def _pred_mask_to_pixel_mask(
        self, pred_mask: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        B = pred_mask.shape[0]
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE
        token_grid = pred_mask.reshape(B, 1, H_tokens, W_tokens)
        return torch.nn.functional.interpolate(
            token_grid, size=(H, W), mode="nearest"
        )

    def _get_sam_weight(self, epoch: int) -> float:
        """Compute current SAM weight with linear ramp."""
        cfg = self.config.model_config_
        sam_max = cfg.sam_weight
        ramp_epochs = cfg.sam_ramp_epochs
        if ramp_epochs <= 0:
            return sam_max
        return sam_max * min(1.0, epoch / ramp_epochs)

    def compute_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        Combined L1 + SAM reconstruction loss with MAE token removal.

        L_total = L1 + λ(t) * SAM

        where λ(t) ramps linearly from 0 to sam_weight over sam_ramp_epochs.
        """
        cfg = self.config.model_config_

        pixels = batch["pixels.npy"].to(self.device)  # (B, 165, H, W)
        mask = self._build_mask(batch)                  # (B, 1, H, W)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        _, _, H, W = pixels.shape

        # Token-level masks
        keep_mask, pred_mask = self._build_prediction_mask(mask, cfg.mask_ratio)

        # Forward pass
        x_hat = model(pixels, mask=mask, keep_mask=keep_mask)

        # Pixel-level prediction mask
        pixel_pred_mask = self._pred_mask_to_pixel_mask(pred_mask, H, W)

        # Erode validity mask
        eroded_mask = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)

        # Loss mask: prediction targets AND valid AND interior
        loss_mask = pixel_pred_mask * eroded_mask  # (B, 1, H, W)

        # --- L1 loss ---
        per_pixel_l1 = (x_hat - pixels).abs()  # (B, 165, H, W)
        # Mean across spectral bands, then apply spatial mask
        per_pixel_l1_mean = per_pixel_l1.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        valid_l1 = per_pixel_l1_mean[loss_mask == 1]

        if valid_l1.numel() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        # --- SAM loss (always computed for logging, only weighted into loss when λ > 0) ---
        sam_weight = self._get_sam_weight(self._current_epoch)
        sam_loss = self._sam_loss(x_hat, pixels, loss_mask)

        # --- Combined loss with optional trimming ---
        l1_loss = valid_l1.mean()

        if cfg.trim_fraction > 0:
            per_pixel_sam = self._per_pixel_sam(x_hat, pixels, loss_mask)
            per_pixel_combined = valid_l1 + sam_weight * per_pixel_sam
            num_keep = int(per_pixel_combined.numel() * (1 - cfg.trim_fraction))
            if num_keep > 0:
                trimmed, _ = per_pixel_combined.sort()
                loss = trimmed[:num_keep].mean()
            else:
                loss = per_pixel_combined.mean()
        else:
            loss = l1_loss + sam_weight * sam_loss

        # Accumulate component losses for epoch-level logging
        self._epoch_l1_sum += l1_loss.item() * num_kept
        self._epoch_sam_sum += sam_loss.item() * num_kept
        self._epoch_loss_samples += num_kept

        return loss, num_kept

    def _per_pixel_sam(
        self, x_hat: torch.Tensor, x: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-pixel SAM values at masked positions (for trimming).
        Uses atan2 formulation to avoid arccos gradient instability."""
        eps = 1e-8
        x_m = x * mask
        xh_m = x_hat * mask
        dot = (xh_m * x_m).sum(dim=1, keepdim=True)
        norm_hat = (xh_m * xh_m).sum(dim=1, keepdim=True).sqrt()
        norm_x = (x_m * x_m).sum(dim=1, keepdim=True).sqrt()
        cross_norm = (norm_hat * norm_x).clamp(min=eps)
        cos_term = dot.clamp(-cross_norm, cross_norm)
        sin_term = (cross_norm ** 2 - cos_term ** 2).clamp(min=0).sqrt()
        angles = torch.atan2(sin_term, cos_term)
        return angles[mask == 1]

    def compute_validation_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """Two-pass random masking validation, matching training regime."""
        cfg = self.config.model_config_
        pixels = batch["pixels.npy"].to(self.device)
        mask = self._build_mask(batch)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        _, _, H, W = pixels.shape
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE
        N = H_tokens * W_tokens

        token_validity = TokenMasking.pixel_mask_to_token_mask(
            mask,
            kernel_size=STAGE1_KERNEL_SIZE,
            stride=STAGE1_STRIDE,
            padding=STAGE1_PADDING,
        )

        rand_mask = (torch.rand(1, N, device=self.device) > 0.5).float()

        # Pass 1
        pred_mask_1 = token_validity * (1.0 - rand_mask)
        keep_mask_1 = 1.0 - pred_mask_1

        # Pass 2
        pred_mask_2 = token_validity * rand_mask
        keep_mask_2 = 1.0 - pred_mask_2

        B = pred_mask_1.shape[0]
        pass1_pixels = pred_mask_1.reshape(B, 1, H_tokens, W_tokens)
        pass1_pixels = torch.nn.functional.interpolate(pass1_pixels, size=(H, W), mode="nearest")
        pass2_pixels = pred_mask_2.reshape(B, 1, H_tokens, W_tokens)
        pass2_pixels = torch.nn.functional.interpolate(pass2_pixels, size=(H, W), mode="nearest")

        x_hat_1 = model(pixels, mask=mask, keep_mask=keep_mask_1)
        x_hat_2 = model(pixels, mask=mask, keep_mask=keep_mask_2)

        x_hat = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels

        eroded_mask = TokenMasking.erode_mask(mask, kernel_size=cfg.erosion_kernel_size)

        # L1 component
        l1 = ((x_hat - pixels).abs().mean(dim=1, keepdim=True) * eroded_mask).sum() / eroded_mask.sum().clamp(min=1)

        # SAM component
        sam_weight = self._get_sam_weight(self._current_epoch)
        sam = self._sam_loss(x_hat, pixels, eroded_mask) if sam_weight > 0 else torch.tensor(0.0, device=self.device)

        loss = l1 + sam_weight * sam

        return loss, num_kept

    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        loss, num_kept = self.compute_validation_loss(batch, model)
        return loss.item(), num_kept
