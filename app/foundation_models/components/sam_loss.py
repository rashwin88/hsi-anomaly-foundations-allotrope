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
        # Broadcast mask to spectral dimension for element-wise ops
        # mask: (B, 1, H, W) — same for all bands

        # Dot product along spectral axis: sum over C
        dot = (x_hat * x * mask).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        # Norms along spectral axis
        norm_hat = (x_hat * x_hat * mask).sum(dim=1, keepdim=True).sqrt()  # (B, 1, H, W)
        norm_x = (x * x * mask).sum(dim=1, keepdim=True).sqrt()            # (B, 1, H, W)

        # Cosine similarity, clamped to [-1, 1] for numerical safety
        cos_sim = dot / (norm_hat * norm_x + self.eps)
        cos_sim = cos_sim.clamp(-1.0, 1.0)

        # Spectral angle in radians
        angles = torch.acos(cos_sim)  # (B, 1, H, W)

        # Spatial mask: which pixels to include
        spatial_mask = mask[:, 0:1, :, :]  # (B, 1, H, W) — already correct shape

        # Mean angle over masked positions
        num_valid = spatial_mask.sum().clamp(min=1)
        sam_loss = (angles * spatial_mask).sum() / num_valid

        return sam_loss
