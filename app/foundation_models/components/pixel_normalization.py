"""
Per-band z-score normalisation layers used by every model that has one.

Wrapped by SpatialAutoencoder, NormalizedMaskedSpatialAutoencoder, SegFormerMAE
and HyperspectralSegFormerMAE. Stats come from app/constants/*.json at training
time, or from an InferenceConfig.pixel_stats_override at inference time.

Why this matters more than it looks: the stats are `register_buffer`s, not plain
attributes, so they are part of the module's state_dict and travel INSIDE the
checkpoint .pt file. You cannot swap normalisation stats without rebuilding the
model - which is exactly why PixelStatsOverride exists for uncalibrated sensors
whose values sit nowhere near the baked-in distribution.

Shapes: mean/std are stored (1, C, 1, 1) so they broadcast against (B, C, H, W).
"""

import torch
import torch.nn as nn

class PixelNormalize(nn.Module):
    """Normalizes input: (x - mean) / std. Stats stored as registered buffers."""

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        # (1, C, 1, 1) for broadcasting against (B, C, H, W)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

class PixelDenormalize(nn.Module):
    """Inverts normalization: x * std + mean."""

    def __init__(self, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.std + self.mean


class WavelengthStats(nn.Module):
    """Mean and std as curves over wavelength, readable at any band list.

    For models whose input is not tied to fixed band positions (the cross-attention stem,
    docs/proposals/spectral-attention-bottleneck.md decision 5): the curve is measured once,
    on SpectralEarth's train split (stats.json beside the shards), and each sensor reads it
    at its own wavelengths by linear interpolation. Outside the measured range the end
    values are held. Buffers, so the curve travels inside the checkpoint like PixelNormalize's.
    """

    def __init__(self, wavelengths_nm: list[float], mean: list[float], std: list[float]):
        super().__init__()
        # SpectralEarth's band order is not monotonic in wavelength (detector overlap near
        # 900-1000 nm); interpolation needs the curve sorted.
        order = torch.argsort(torch.tensor(wavelengths_nm, dtype=torch.float64))
        for name, v in (("wl", wavelengths_nm), ("mean", mean), ("std", std)):
            self.register_buffer(name, torch.tensor(v, dtype=torch.float32)[order])

    def at(self, wavelengths_nm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N,) or (B, N) wavelengths -> (mean, std), each shaped like the input."""
        q = wavelengths_nm.float().clamp(self.wl[0], self.wl[-1])
        hi = torch.searchsorted(self.wl, q.contiguous()).clamp(1, len(self.wl) - 1)
        lo = hi - 1
        t = ((q - self.wl[lo]) / (self.wl[hi] - self.wl[lo]).clamp_min(1e-6)).clamp(0, 1)
        return (torch.lerp(self.mean[lo], self.mean[hi], t), torch.lerp(self.std[lo], self.std[hi], t))

    @staticmethod
    def _bcast(v: torch.Tensor) -> torch.Tensor:
        return v.view(1, -1, 1, 1) if v.dim() == 1 else v[:, :, None, None]

    def normalize(self, x: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        """(B, N, H, W) reflectance -> z-scores at those wavelengths."""
        mean, std = self.at(wavelengths_nm)
        return (x - self._bcast(mean)) / self._bcast(std)

    def denormalize(self, x: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        mean, std = self.at(wavelengths_nm)
        return x * self._bcast(std) + self._bcast(mean)
