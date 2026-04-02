"""
Trainer for the SegFormer MAE reconstruction model.

Implements MAE-style training with true token removal:
    1. Convert pixel validity mask to token-level validity
    2. From valid tokens, randomly select mask_ratio fraction as prediction targets
    3. Remove prediction targets from the token sequence (encoder never sees them)
    4. Compute L1 loss only at prediction target pixel positions

The model receives the full image (all pixels), but the encoder's Stage 1
physically removes prediction tokens after patch embedding. The encoder
processes only visible + invalid tokens. After encoding, tokens are scattered
back to the full grid (zeros at prediction positions) and the decoder
reconstructs the full image. Loss is computed only where tokens were masked.

Patches with less than 40% valid pixels are discarded from the batch.
"""

import json
import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_trainer import FoundationTrainer
from app.foundation_models.components.seg_former_mae import SegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
from app.models.training.training_config import SegFormerMAEConfig

logger = logging.getLogger("SegFormerMAETrainer")

MIN_VALID_PIXEL_FRACTION = 0.4

# Stage 1 patch embedding parameters (must match SegFormerEncoder)
STAGE1_KERNEL_SIZE = 7
STAGE1_STRIDE = 4


class SegFormerMAETrainer(FoundationTrainer):

    def build_model(self) -> nn.Module:
        cfg: SegFormerMAEConfig = self.config.model_config_
        pixel_mean, pixel_std = None, None
        stats_path = self.config.data.pixel_stats_path
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
            pixel_mean = stats["mean"]
            pixel_std = stats["std"]
            logger.info(f"Pixel normalization stats loaded: mean={pixel_mean}, std={pixel_std}")
        return SegFormerMAE(
            in_channels=cfg.in_channels,
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

    def _build_prediction_mask(
        self, pixel_mask: torch.Tensor, mask_ratio: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate token-level keep and prediction masks from pixel-level validity.

        Uses the same kernel_size and stride as Stage 1's OverlapPatchEmbedding
        so token validity matches what OPE actually computed.

        Args:
            pixel_mask: (B, 1, H, W) -- pixel validity
            mask_ratio: fraction of valid tokens to mask

        Returns:
            keep_mask: (B, N) -- 1=keep in encoder, 0=remove
            pred_mask: (B, N) -- 1=prediction target, 0=everything else
        """
        # Convert pixel validity to token validity
        # Uses same kernel and stride as OPE Stage 1
        token_mask = TokenMasking.pixel_mask_to_token_mask(
            pixel_mask,
            kernel_size=STAGE1_KERNEL_SIZE,
            stride=STAGE1_STRIDE,
        )

        # From valid tokens, randomly select mask_ratio as prediction targets
        keep_mask, pred_mask = TokenMasking.generate_prediction_mask(
            token_mask, mask_ratio, self.device
        )

        return keep_mask, pred_mask

    def _pred_mask_to_pixel_mask(self, pred_mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Convert token-level prediction mask to pixel-level mask for loss computation.

        Each token covers a stride x stride pixel block. Expanding the token mask
        to pixel resolution gives us a mask of which pixels to compute loss on.

        Args:
            pred_mask: (B, N) -- 1=prediction target token
            H, W:     original pixel dimensions

        Returns:
            pixel_pred_mask: (B, 1, H, W) -- 1 at prediction pixel positions
        """
        B = pred_mask.shape[0]
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE

        # Reshape to 2D token grid: (B, N) -> (B, 1, H_tokens, W_tokens)
        token_grid = pred_mask.reshape(B, 1, H_tokens, W_tokens)

        # Upsample to pixel resolution via nearest-neighbor
        # Each token's value is replicated across its stride x stride pixel block
        # (B, 1, H_tokens, W_tokens) -> (B, 1, H, W)
        pixel_pred_mask = torch.nn.functional.interpolate(
            token_grid, size=(H, W), mode='nearest'
        )

        return pixel_pred_mask

    def compute_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        MAE reconstruction loss with true token removal.

        1. Build pixel validity mask
        2. Generate token-level prediction mask (50% of valid tokens)
        3. Forward pass with keep_mask (encoder removes prediction tokens)
        4. L1 loss only at prediction target pixel positions

        Loss = sum(|x_hat - x| * pred_pixel_mask) / sum(pred_pixel_mask)
        """
        cfg: SegFormerMAEConfig = self.config.model_config_

        pixels = batch["pixels.npy"].to(self.device)  # (B, C, H, W)
        mask = self._build_mask(batch)                  # (B, 1, H, W)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        _, _, H, W = pixels.shape

        # Generate token-level masks
        keep_mask, pred_mask = self._build_prediction_mask(mask, cfg.mask_ratio)

        # Forward: model applies pixel masking + normalization internally,
        # encoder removes prediction tokens at Stage 1
        x_hat = model(pixels, mask=mask, keep_mask=keep_mask)
        # x_hat: (B, 1, H, W) -- full reconstruction in temperature space

        # Convert token prediction mask to pixel-level for loss
        pixel_pred_mask = self._pred_mask_to_pixel_mask(pred_mask, H, W)

        # Also intersect with pixel validity -- only compute loss on
        # pixels that are both prediction targets AND valid
        loss_mask = pixel_pred_mask * mask

        # L1 loss only at masked valid pixel positions
        loss = ((x_hat - pixels).abs() * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        return loss, num_kept

    def _checkerboard_keep_mask(
        self, pixel_mask: torch.Tensor, invert: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build token-level keep/pred masks from a checkerboard pattern.
        Used for validation to match inference regime.

        Args:
            pixel_mask: (B, 1, H, W) pixel validity
            invert:     False for pass 1, True for pass 2

        Returns:
            keep_mask: (B, N) -- 1=keep, 0=remove
            pred_mask: (B, N) -- 1=prediction target
        """
        _, _, H, W = pixel_mask.shape
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE

        checker = TokenMasking.checkerboard_token_mask(
            H_tokens, W_tokens, cell_size=1,
            device=self.device, invert=invert,
        )
        token_validity = TokenMasking.pixel_mask_to_token_mask(
            pixel_mask, kernel_size=STAGE1_KERNEL_SIZE, stride=STAGE1_STRIDE
        )
        pred_mask = token_validity * (1.0 - checker)
        keep_mask = 1.0 - pred_mask
        return keep_mask, pred_mask

    def compute_validation_loss(
        self, batch: dict, model: nn.Module
    ) -> tuple[torch.Tensor, int]:
        """
        Validation loss using two-pass checkerboard masking.

        Matches the inference regime: each pixel is reconstructed from context
        only (never from itself). Loss is computed on all valid pixels across
        both passes, combining each pixel's value from the pass where its
        token was masked.

        This gives a val loss that tracks the actual inference performance,
        unlike full reconstruction which is a different task than what the
        model is trained for.
        """
        pixels = batch["pixels.npy"].to(self.device)
        mask = self._build_mask(batch)

        pixels, mask, num_kept = self._filter_batch(pixels, mask)
        if num_kept == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0

        _, _, H, W = pixels.shape
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE

        # Checkerboard at token level for combining passes
        checker = TokenMasking.checkerboard_token_mask(
            H_tokens, W_tokens, cell_size=1, device=self.device, invert=False,
        )
        # Expand to pixel level: (1, N) -> (1, 1, H, W)
        checker_pixels = checker.reshape(1, 1, H_tokens, W_tokens)
        checker_pixels = torch.nn.functional.interpolate(
            checker_pixels, size=(H, W), mode='nearest'
        )
        checker_inv_pixels = 1.0 - checker_pixels

        # Pass 1: mask "black" tokens, reconstruct
        keep_mask_1, _ = self._checkerboard_keep_mask(mask, invert=False)
        x_hat_1 = model(pixels, mask=mask, keep_mask=keep_mask_1)

        # Pass 2: mask "white" tokens, reconstruct
        keep_mask_2, _ = self._checkerboard_keep_mask(mask, invert=True)
        x_hat_2 = model(pixels, mask=mask, keep_mask=keep_mask_2)

        # Combine: each pixel from the pass where its token was masked
        x_hat = x_hat_1 * checker_inv_pixels + x_hat_2 * checker_pixels

        # L1 loss on all valid pixels
        loss = ((x_hat - pixels).abs() * mask).sum() / mask.sum().clamp(min=1)

        return loss, num_kept

    def validation_step(
        self, batch: dict, model: nn.Module
    ) -> tuple[float, int]:
        loss, num_kept = self.compute_validation_loss(batch, model)
        return loss.item(), num_kept
