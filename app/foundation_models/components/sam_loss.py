"""
Spectral Angle Mapper (SAM) loss for hyperspectral reconstruction.

Measures the angular distance between predicted and target spectral
vectors at each pixel. Penalises spectral shape distortion independently
of magnitude — a spectrum can be perfectly bright but have the wrong
shape, and SAM will catch it.

SAM(a, b) = arccos( (a · b) / (||a|| · ||b|| + ε) )

Values in radians: 0 = identical shape, π/2 = orthogonal.
Typical reconstruction SAM: 0.01–0.10 rad (0.6°–5.7°).
"""

import torch
import torch.nn as nn


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper loss.

    Computes the mean spectral angle between predicted and target spectra
    at positions indicated by a mask.

    Args:
        eps: Small constant for numerical stability in the denominator.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_hat: (B, C, H, W) — predicted spectra
            x:     (B, C, H, W) — target spectra
            mask:  (B, 1, H, W) — 1 at positions to compute loss, 0 elsewhere

        Returns:
            Scalar SAM loss (mean over masked positions), in radians.
        """
        # Dot product along spectral axis: sum over C
        # Mask broadcasts (B,1,H,W) → (B,C,H,W)
        x_m = x * mask
        xh_m = x_hat * mask

        dot = (xh_m * x_m).sum(dim=1, keepdim=True)           # (B, 1, H, W)
        norm_hat = (xh_m * xh_m).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
        norm_x = (x_m * x_m).sum(dim=1, keepdim=True).sqrt()      # (B, 1, H, W)

        # Cross product magnitude for atan2 formulation:
        # ||a × b|| = ||a|| ||b|| sin(θ)
        # a · b     = ||a|| ||b|| cos(θ)
        # θ = atan2(sin_term, cos_term)
        #
        # This avoids arccos which has infinite gradient at ±1
        # and is numerically stable everywhere.
        cross_norm = (norm_hat * norm_x).clamp(min=self.eps)
        cos_term = dot.clamp(-cross_norm, cross_norm)
        sin_term = (cross_norm ** 2 - cos_term ** 2).clamp(min=0).sqrt()

        angles = torch.atan2(sin_term, cos_term)  # (B, 1, H, W), always in [0, π]

        # Spatial mask
        spatial_mask = mask[:, 0:1, :, :]

        # Mean angle over masked positions
        num_valid = spatial_mask.sum().clamp(min=1)
        sam_loss = (angles * spatial_mask).sum() / num_valid

        return sam_loss
