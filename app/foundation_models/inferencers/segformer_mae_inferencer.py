"""
Inferencer for the SegFormer MAE reconstruction model.

Uses two-pass checkerboard token masking for context-only reconstruction:

    Pass 1: Mask "black" checkerboard tokens, encode visible "white" tokens,
            reconstruct the full image -> take prediction at "black" positions.

    Pass 2: Mask "white" checkerboard tokens, encode visible "black" tokens,
            reconstruct the full image -> take prediction at "white" positions.

    Combine: Each pixel's final value comes from the pass where its token
             was masked. Every pixel is predicted from context, never from itself.

This matches the training regime (50% token masking) and ensures the
reconstruction error at each pixel reflects genuine prediction difficulty,
not trivial self-reconstruction.

For full-scene inference, patches are extracted via PatchPlanGenerator
with a sliding window, reconstructed independently, and overlap-averaged
back into the full frame.
"""

import json
import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.components.seg_former_mae import SegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
from app.models.patches.patching_request import PatchRequest
from app.models.training.training_config import SegFormerMAEConfig
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

logger = logging.getLogger("SegFormerMAEInferencer")

# Stage 1 patch embedding parameters (must match SegFormerEncoder)
STAGE1_KERNEL_SIZE = 7
STAGE1_STRIDE = 4


class SegFormerMAEInferencer(FoundationInferencer):

    def build_model(self) -> nn.Module:
        cfg: SegFormerMAEConfig = self.config.model_config_
        pixel_mean, pixel_std = None, None
        stats_path = self.config.pixel_stats_path
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
            drop_rate=0.0,  # No dropout at inference
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )

    def _checkerboard_keep_mask(
        self, H: int, W: int, mask: torch.Tensor, invert: bool
    ) -> torch.Tensor:
        """
        Build a token-level keep mask from a checkerboard pattern.

        Combines the checkerboard (which tokens to mask for prediction) with
        the validity mask (which tokens are invalid). Invalid tokens are always
        kept (they carry the zero-signal), only valid checkerboard-selected
        tokens are removed.

        Args:
            H, W:   pixel dimensions of the patch
            mask:   (B, 1, H, W) pixel validity mask
            invert: False for pass 1, True for pass 2

        Returns:
            keep_mask: (B, N) -- 1=keep, 0=remove (prediction target)
        """
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE

        # Checkerboard at token level: 1=visible, 0=masked
        # (1, N) -- broadcastable over batch
        checker = TokenMasking.checkerboard_token_mask(
            H_tokens, W_tokens,
            cell_size=self.config.checkerboard_cell_size,
            device=self.device,
            invert=invert,
        )
        # checker: (1, N) -- 1 = visible in this pass, 0 = prediction target

        # Token validity: convert pixel mask to token mask
        token_validity = TokenMasking.pixel_mask_to_token_mask(
            mask, kernel_size=STAGE1_KERNEL_SIZE, stride=STAGE1_STRIDE
        )
        # token_validity: (B, N) -- 1=valid, 0=invalid

        # Prediction targets: tokens that are both valid AND checkerboard-masked
        # pred_mask = 1 where token is valid and checker says "mask this token"
        pred_mask = token_validity * (1.0 - checker)

        # Keep mask: everything that is NOT a prediction target
        # Keeps: visible valid tokens + all invalid tokens
        keep_mask = 1.0 - pred_mask

        return keep_mask

    def infer(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Two-pass checkerboard reconstruction for a batch of patches.

        Pass 1: Remove "black" checkerboard tokens, reconstruct -> take "black" pixels
        Pass 2: Remove "white" checkerboard tokens, reconstruct -> take "white" pixels
        Combine: Each pixel comes from the pass where its token was masked.

        Args:
            tensor: (B, C, H, W) input patches, already on device.
            mask:   (B, 1, H, W) validity mask, already on device.

        Returns:
            (B, C, H, W) reconstruction where every pixel was predicted
            from context (never from itself). Invalid pixels are zeroed.
        """
        _, _, H, W = tensor.shape
        H_tokens = H // STAGE1_STRIDE
        W_tokens = W // STAGE1_STRIDE

        # Get checkerboard pattern at token level for combining passes
        # (1, N) -- 1 at "white" positions, 0 at "black" positions
        checker = TokenMasking.checkerboard_token_mask(
            H_tokens, W_tokens,
            cell_size=self.config.checkerboard_cell_size,
            device=self.device,
            invert=False,
        )
        checker_inv = 1.0 - checker

        # Expand to pixel level for combining reconstructions
        # (1, N) -> (1, 1, H_tokens, W_tokens) -> upsample -> (1, 1, H, W)
        B_check = checker.shape[0]
        checker_pixels = checker.reshape(B_check, 1, H_tokens, W_tokens)
        checker_pixels = torch.nn.functional.interpolate(
            checker_pixels, size=(H, W), mode='nearest'
        )
        # checker_pixels: (1, 1, H, W) -- 1 at "white" pixel positions
        checker_inv_pixels = 1.0 - checker_pixels

        # --- Pass 1: mask "black" tokens (checker=0), encode "white" tokens ---
        # keep_mask_1: keeps "white" valid + all invalid, removes "black" valid
        keep_mask_1 = self._checkerboard_keep_mask(H, W, mask, invert=False)
        x_hat_1 = self.model(tensor, mask=mask, keep_mask=keep_mask_1)
        # x_hat_1: (B, C, H, W) -- reconstruction, good at "black" (predicted) positions

        # --- Pass 2: mask "white" tokens (checker=1), encode "black" tokens ---
        keep_mask_2 = self._checkerboard_keep_mask(H, W, mask, invert=True)
        x_hat_2 = self.model(tensor, mask=mask, keep_mask=keep_mask_2)
        # x_hat_2: (B, C, H, W) -- reconstruction, good at "white" (predicted) positions

        # --- Combine: each pixel from the pass where its token was MASKED ---
        # Pass 1 masked "black" tokens -> use x_hat_1 at "black" pixel positions
        # Pass 2 masked "white" tokens -> use x_hat_2 at "white" pixel positions
        reconstruction = x_hat_1 * checker_inv_pixels + x_hat_2 * checker_pixels

        # Zero out invalid pixels
        reconstruction = reconstruction * mask

        return reconstruction

    def predict_full_scene(
        self, scene: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding-window checkerboard reconstruction over a full scene.

        Extracts overlapping patches via PatchPlanGenerator, reconstructs
        each with two-pass checkerboard masking, and overlap-averages back
        into the full frame.

        Args:
            scene: (C, H, W) full scene tensor.
            mask:  (1, H, W) validity mask for the full scene.

        Returns:
            (C, H, W) per-pixel reconstruction, overlap-averaged.
        """
        scene = scene.to(self.device)
        mask = mask.to(self.device)

        c, h, w = scene.shape
        ps = self.config.patch_size
        stride = self.config.stride or ps // 2

        request = PatchRequest(
            input_cube=(c, h, w),
            width=ps,
            height=ps,
            stride=stride,
        )
        plan = PatchPlanGenerator().generate_patching_plan(request)

        recon_sum = torch.zeros(c, h, w, device=self.device)
        count = torch.zeros(1, h, w, device=self.device)

        for r, c_start in plan.patch_coordinates:
            patch = scene[:, r:r + ps, c_start:c_start + ps].unsqueeze(0)
            patch_mask = mask[:, r:r + ps, c_start:c_start + ps].unsqueeze(0)

            recon = self.predict(patch, patch_mask)  # (1, C, ps, ps)

            recon_sum[:, r:r + ps, c_start:c_start + ps] += recon.squeeze(0)
            count[:, r:r + ps, c_start:c_start + ps] += patch_mask.squeeze(0)

        reconstruction = torch.where(count > 0, recon_sum / count, recon_sum)

        return reconstruction
