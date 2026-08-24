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
import torch.nn.functional as F

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.components.seg_former_mae import SegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
from app.models.patches.patching_request import PatchRequest
from app.models.training.training_config import SegFormerMAEConfig
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

logger = logging.getLogger("SegFormerMAEInferencer")

# Stage 1 patch embedding parameters (must match SegFormerEncoder)
STAGE1_KERNEL_SIZE = 4
STAGE1_STRIDE = 4
STAGE1_PADDING = 0  # Non-overlapping: kernel=stride, no padding


class SegFormerMAEInferencer(FoundationInferencer):

    def build_model(self) -> nn.Module:
        cfg: SegFormerMAEConfig = self.config.model_config_
        pixel_mean, pixel_std = None, None
        override = self.config.pixel_stats_override
        stats_path = self.config.pixel_stats_path
        if override is not None:
            # Per-scene stats replace the checkpoint's baked-in values -
            # see PixelStatsOverride for why uncalibrated sensors need this.
            pixel_mean, pixel_std = override.mean, override.std
            logger.info("Pixel stats overridden per-scene (source=%s)", override.source)
        elif stats_path is not None:
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
            mask, kernel_size=STAGE1_KERNEL_SIZE, stride=STAGE1_STRIDE,
            padding=STAGE1_PADDING,
        )
        # token_validity: (B, N) -- 1=valid, 0=invalid

        # Prediction targets: tokens that are both valid AND checkerboard-masked
        # pred_mask = 1 where token is valid and checker says "mask this token"
        pred_mask = token_validity * (1.0 - checker)

        # Keep mask: everything that is NOT a prediction target
        # Keeps: visible valid tokens + all invalid tokens
        keep_mask = 1.0 - pred_mask

        return keep_mask

    def _token_mask_to_pixel_mask(self, token_mask, H_tokens, W_tokens, H, W):
        """Expand a (B, N) or (1, N) token mask to (B, 1, H, W) pixel mask."""
        B = token_mask.shape[0]
        pixel_mask = token_mask.reshape(B, 1, H_tokens, W_tokens)
        pixel_mask = F.interpolate(pixel_mask, size=(H, W), mode='nearest')
        return pixel_mask

    def _build_random_keep_mask(self, H_tokens, W_tokens, mask):
        """
        Build a random 50% token mask and its complement for two-pass inference.
        No systematic grid pattern — artifacts are randomly distributed and
        cancel out when averaged across overlapping patches.

        Args:
            H_tokens, W_tokens: token grid dimensions
            mask: (B, 1, H, W) pixel validity mask

        Returns:
            keep_mask_1: (B, N) -- pass 1: keep these, mask the rest
            keep_mask_2: (B, N) -- pass 2: exact complement of pass 1
        """
        N = H_tokens * W_tokens

        # Token validity
        token_validity = TokenMasking.pixel_mask_to_token_mask(
            mask, kernel_size=STAGE1_KERNEL_SIZE, stride=STAGE1_STRIDE,
            padding=STAGE1_PADDING,
        )
        # token_validity: (B, N)

        # Random 50% split of valid tokens
        # rand_mask: 1=visible in pass 1, 0=masked in pass 1
        rand_mask = (torch.rand(1, N, device=self.device) > 0.5).float()

        # Pass 1: prediction targets are valid tokens where rand_mask=0
        pred_mask_1 = token_validity * (1.0 - rand_mask)
        keep_mask_1 = 1.0 - pred_mask_1

        # Pass 2: exact complement — prediction targets are valid tokens where rand_mask=1
        pred_mask_2 = token_validity * rand_mask
        keep_mask_2 = 1.0 - pred_mask_2

        return keep_mask_1, keep_mask_2, rand_mask

    def infer(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Two-pass reconstruction for a batch of patches.

        Supports two masking strategies (configured via inference_config.masking_strategy):
            'checkerboard': deterministic token-level checkerboard pattern
            'random': random 50% mask with complement for second pass

        Both strategies ensure every pixel is predicted from context (never from itself).
        Random masking avoids the systematic grid artifacts that checkerboard produces
        at token boundaries.

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
        strategy = self.config.masking_strategy

        if strategy == "checkerboard":
            # Deterministic checkerboard at token level
            checker = TokenMasking.checkerboard_token_mask(
                H_tokens, W_tokens,
                cell_size=self.config.checkerboard_cell_size,
                device=self.device,
                invert=False,
            )
            pass1_pixels = self._token_mask_to_pixel_mask(
                1.0 - checker, H_tokens, W_tokens, H, W
            )
            pass2_pixels = self._token_mask_to_pixel_mask(
                checker, H_tokens, W_tokens, H, W
            )

            keep_mask_1 = self._checkerboard_keep_mask(H, W, mask, invert=False)
            keep_mask_2 = self._checkerboard_keep_mask(H, W, mask, invert=True)

        elif strategy == "random":
            # Random 50% mask with complement
            keep_mask_1, keep_mask_2, rand_mask = self._build_random_keep_mask(
                H_tokens, W_tokens, mask
            )
            # pass1 masked tokens (rand_mask=0) → take pass1 reconstruction there
            pass1_pixels = self._token_mask_to_pixel_mask(
                1.0 - rand_mask, H_tokens, W_tokens, H, W
            )
            # pass2 masked tokens (rand_mask=1) → take pass2 reconstruction there
            pass2_pixels = self._token_mask_to_pixel_mask(
                rand_mask, H_tokens, W_tokens, H, W
            )

        else:
            raise ValueError(f"Unknown masking strategy: {strategy}")

        # --- Pass 1 ---
        x_hat_1 = self.model(tensor, mask=mask, keep_mask=keep_mask_1)

        # --- Pass 2 ---
        x_hat_2 = self.model(tensor, mask=mask, keep_mask=keep_mask_2)

        # --- Combine: each pixel from the pass where its token was masked ---
        reconstruction = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels

        # Zero out invalid pixels
        reconstruction = reconstruction * mask

        return reconstruction

    def predict_full_scene(
        self, scene: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Batched sliding-window reconstruction over a full scene.

        Extracts overlapping patches via PatchPlanGenerator, batches them
        for efficient GPU utilization, reconstructs with two-pass masking,
        and overlap-averages back into the full frame.

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
        batch_size = self.config.inference_batch_size

        # Erode the validity mask to exclude border pixels whose reconstructions
        # are unreliable. Uses the configurable erosion_kernel_size which should
        # be >= OPE kernel size to cover the full receptive field overlap.
        erosion_ks = self.config.erosion_kernel_size
        eroded_mask = TokenMasking.erode_mask(
            mask.unsqueeze(0), kernel_size=erosion_ks
        ).squeeze(0)

        request = PatchRequest(
            input_cube=(c, h, w),
            width=ps,
            height=ps,
            stride=stride,
        )
        plan = PatchPlanGenerator().generate_patching_plan(request)
        coords = plan.patch_coordinates

        recon_sum = torch.zeros(c, h, w, device=self.device)
        count = torch.zeros(1, h, w, device=self.device)

        # Minimum valid pixel fraction to process a patch.
        # Matches training: patches with < 40% valid pixels were discarded.
        # The model never learned to reconstruct mostly-invalid patches,
        # so running inference on them produces garbage at boundaries.
        MIN_VALID_FRACTION = 0.1

        # Process patches in batches for GPU efficiency
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]

            # Extract patches and their masks
            all_patches = [scene[:, r:r + ps, cs:cs + ps] for r, cs in batch_coords]
            all_masks = [mask[:, r:r + ps, cs:cs + ps] for r, cs in batch_coords]

            # Filter: skip patches below the validity threshold
            # Same criterion as training — model never saw mostly-invalid patches
            valid_indices = []
            for i, m in enumerate(all_masks):
                valid_frac = m.float().mean().item()
                if valid_frac >= MIN_VALID_FRACTION:
                    valid_indices.append(i)

            if len(valid_indices) == 0:
                continue

            # Stack only the valid patches into a batch
            patches = torch.stack([all_patches[i] for i in valid_indices])
            patch_masks = torch.stack([all_masks[i] for i in valid_indices])

            # Two-pass reconstruction on the whole batch at once
            recon = self.predict(patches, patch_masks)  # (B_valid, C, ps, ps)

            # Scatter results back into the full scene
            for j, idx in enumerate(valid_indices):
                r, cs = batch_coords[idx]
                patch_eroded = eroded_mask[:, r:r + ps, cs:cs + ps]
                recon_sum[:, r:r + ps, cs:cs + ps] += recon[j] * patch_eroded
                count[:, r:r + ps, cs:cs + ps] += patch_eroded

        # Where count > 0: average the overlapping reconstructions
        # Where count == 0: no valid reconstruction — use the original scene value
        # so these pixels have zero residual and don't create false detections
        reconstruction = torch.where(count > 0, recon_sum / count, scene)

        return reconstruction
