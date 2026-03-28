"""
Concrete inferencer for the SpatialAutoencoder.

Implements checkerboard-masked reconstruction: for each patch, alternating
cells are nulled before the forward pass. The model reconstructs the
nulled cells from context. Two passes (checkerboard + inverted) ensure
every pixel gets a reconstruction from context.

For full-scene inference, patches are extracted via PatchPlanGenerator
with a sliding window, reconstructed independently, and overlap-averaged
back into the full frame.
"""

import logging

import torch
import torch.nn as nn

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.components.unnormalized_spatial_auto_encoder import UnNormalizedSpatialAutoencoder
from app.models.patches.patching_request import PatchRequest
from app.models.training.training_config import SpatialMaskedAutoEncoderConfig
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

logger = logging.getLogger("SpatialAutoencoderInferencer")


class MaskedSpatialAutoencoderInferencer(FoundationInferencer):

    def build_model(self) -> nn.Module:
        cfg: SpatialMaskedAutoEncoderConfig  = self.config.model_config_
        return UnNormalizedSpatialAutoencoder(
            in_channels=cfg.in_channels,
            base_channels=cfg.base_channels,
            num_stages=cfg.num_stages,
            kernel_size=cfg.kernel_size,
        )

    def _build_checkerboard(self, h: int, w: int, invert: bool = False) -> torch.Tensor:
        """
        Build a checkerboard mask of shape (1, 1, H, W).

        Cell size is controlled by config.checkerboard_cell_size.

        For cell_size=2 on an 8x8 patch:
          rows = [0,0,1,1,2,2,3,3]  (arange // cell_size)
          cols = [0,0,1,1,2,2,3,3]
          grid = (rows[:, None] + cols[None, :]) % 2

          Result:
            0 0 1 1 0 0 1 1
            0 0 1 1 0 0 1 1
            1 1 0 0 1 1 0 0
            1 1 0 0 1 1 0 0
            ...

        When invert=False, 1 = kept, 0 = nulled.
        When invert=True, flipped.
        """
        cell = self.config.checkerboard_cell_size
        rows = torch.arange(h, device=self.device) // cell
        cols = torch.arange(w, device=self.device) // cell
        grid = (rows[:, None] + cols[None, :]) % 2  # (H, W)

        if invert:
            grid = 1 - grid

        return grid.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)

    def infer(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Checkerboard-masked reconstruction for a batch of patches.

        For each patch:
          1. Null checkerboard cells, forward pass, get reconstruction of nulled cells
          2. Null inverted checkerboard cells, forward pass, get reconstruction of those
          3. Combine: each pixel's value comes from the pass where it was nulled

        Invalid pixels (mask=0) are zeroed in the output.

        Args:
            tensor: (B, C, H, W) input patches, already on device.
            mask:   (B, 1, H, W) validity mask, already on device.

        Returns:
            (B, C, H, W) reconstruction where every pixel was reconstructed
            from context (never from itself). Invalid pixels are zeroed.
        """
        _, _, h, w = tensor.shape

        checker = self._build_checkerboard(h, w, invert=False)  # (1,1,H,W)
        checker_inv = 1 - checker

        # Pass 1: keep checker=1 cells visible, null checker=0 cells
        x_hat_1, _ = self.model(tensor, validity_mask=mask, input_mask=checker*mask)

        # Pass 2: keep checker=0 cells visible, null checker=1 cells
        x_hat_2, _ = self.model(tensor, validity_mask=mask, input_mask=checker_inv*mask)

        # Combine: take each pixel's reconstruction from the pass where it was HIDDEN
        # Pass 1 hid checker=0 (checker_inv) cells → use x_hat_1 for those
        # Pass 2 hid checker=1 (checker) cells     → use x_hat_2 for those
        reconstruction = x_hat_1 * checker_inv + x_hat_2 * checker  # (B, C, H, W)

        # Zero out invalid pixels
        reconstruction = reconstruction * mask

        return reconstruction

    def predict_full_scene(
        self, scene: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Sliding-window checkerboard reconstruction over a full scene.

        Extracts overlapping patches via PatchPlanGenerator, reconstructs
        each, and overlap-averages back into the full frame.

        Args:
            scene: (C, H, W) full scene tensor.
            mask:  (1, H, W) validity mask for the full scene.

        Returns:
            (C, H, W) per-pixel reconstruction, overlap-averaged.
        """
        # Move inputs to device upfront
        scene = scene.to(self.device)
        mask = mask.to(self.device)

        c, h, w = scene.shape
        ps = self.config.patch_size
        stride = self.config.stride or ps // 2

        # Build patch plan using existing PatchPlanGenerator
        request = PatchRequest(
            input_cube=(c, h, w),
            width=ps,
            height=ps,
            stride=stride,
        )
        plan = PatchPlanGenerator().generate_patching_plan(request)

        # Accumulators for overlap averaging
        recon_sum = torch.zeros(c, h, w, device=self.device)
        count = torch.zeros(1, h, w, device=self.device)

        for r, c_start in plan.patch_coordinates:
            patch = scene[:, r:r + ps, c_start:c_start + ps].unsqueeze(0)
            patch_mask = mask[:, r:r + ps, c_start:c_start + ps].unsqueeze(0)

            recon = self.predict(patch, patch_mask)  # (1, C, ps, ps)

            recon_sum[:, r:r + ps, c_start:c_start + ps] += recon.squeeze(0)
            count[:, r:r + ps, c_start:c_start + ps] += patch_mask.squeeze(0)

        # Average where we have coverage, zero elsewhere
        reconstruction = torch.where(count > 0, recon_sum / count, recon_sum)

        return reconstruction
