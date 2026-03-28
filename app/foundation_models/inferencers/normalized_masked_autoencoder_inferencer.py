"""
Inferencer for the NormalizedMaskedSpatialAutoencoder.

Uses checkerboard-masked reconstruction with the 3-channel architecture
(normalized pixels + validity_mask + input_mask). Two passes ensure
every pixel is reconstructed from context, never from itself.
"""

import json
import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.components.normalized_masked_spatial_auto_encoder import (
    NormalizedMaskedSpatialAutoencoder,
)
from app.models.patches.patching_request import PatchRequest
from app.models.training.training_config import NormalizedMaskedAutoEncoderConfig
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

logger = logging.getLogger("NormalizedMaskedAutoencoderInferencer")


class NormalizedMaskedAutoencoderInferencer(FoundationInferencer):

    def build_model(self) -> nn.Module:
        cfg: NormalizedMaskedAutoEncoderConfig = self.config.model_config_
        pixel_mean, pixel_std = None, None
        stats_path = self.config.pixel_stats_path
        if stats_path is not None:
            with open(stats_path) as f:
                stats = json.load(f)
            pixel_mean = stats["mean"]
            pixel_std = stats["std"]
            logger.info(f"Pixel normalization stats loaded: mean={pixel_mean}, std={pixel_std}")
        return NormalizedMaskedSpatialAutoencoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            num_stages=cfg.num_stages,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            kernel_size=cfg.kernel_size,
        )

    def _build_checkerboard(self, h: int, w: int, invert: bool = False) -> torch.Tensor:
        """
        Build a checkerboard mask of shape (1, 1, H, W).

        Cell size is controlled by config.checkerboard_cell_size.
        When invert=False, 1 = kept, 0 = nulled.
        """
        cell = self.config.checkerboard_cell_size
        rows = torch.arange(h, device=self.device) // cell
        cols = torch.arange(w, device=self.device) // cell
        grid = (rows[:, None] + cols[None, :]) % 2

        if invert:
            grid = 1 - grid

        return grid.unsqueeze(0).unsqueeze(0).float()

    def infer(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Checkerboard-masked reconstruction for a batch of patches.

        Two passes with complementary checkerboard masks ensure every
        pixel is reconstructed from context (never from itself).
        """
        _, _, h, w = tensor.shape

        checker = self._build_checkerboard(h, w, invert=False)
        checker_inv = 1 - checker

        # Pass 1: keep checker=1 cells visible, null checker=0 cells
        x_hat_1, _ = self.model(tensor, validity_mask=mask, input_mask=checker * mask)

        # Pass 2: keep checker=0 cells visible, null checker=1 cells
        x_hat_2, _ = self.model(tensor, validity_mask=mask, input_mask=checker_inv * mask)

        # Combine: take each pixel from the pass where it was HIDDEN
        # Pass 1 hid checker=0 (checker_inv) cells -> use x_hat_1 for those
        # Pass 2 hid checker=1 (checker) cells     -> use x_hat_2 for those
        reconstruction = x_hat_1 * checker_inv + x_hat_2 * checker

        reconstruction = reconstruction * mask

        return reconstruction

    def predict_full_scene(
        self, scene: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding-window checkerboard reconstruction over a full scene.

        Extracts overlapping patches via PatchPlanGenerator, reconstructs
        each, and overlap-averages back into the full frame.
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

            recon = self.predict(patch, patch_mask)

            recon_sum[:, r:r + ps, c_start:c_start + ps] += recon.squeeze(0)
            count[:, r:r + ps, c_start:c_start + ps] += patch_mask.squeeze(0)

        reconstruction = torch.where(count > 0, recon_sum / count, recon_sum)

        return reconstruction
