"""
Pixel screens for SpectralEarth patches: which pixels are fit to train on, and why not.

Applied by the torch dataset and by SpectralEarthPatch.to_vendable(), where the combined
keep-mask plays the part EnMAP's provider quality masks play in
app/utils/dataset_builder/enmap_dataset_builder.py: rejected pixels are invalidated as a
whole spectral column before gap interpolation and resampling.

SpectralEarth ships no quality layers at all (no cloud, shadow, haze or snow masks), and
DLR filtered clouds only per source scene (< ~10%, by eye), not per patch. Without a
screen, cloud pixels would enter training as "normal" surface. This module is the seam
where screening plugs in; NodataScreen is the first and simplest screen, and a cloud
screen joins the chain once it has been shown to work without the 1380 nm band
SpectralEarth dropped.

Every screen reports how many pixels it rejected, so a training run can say what it threw
away and why rather than just "fewer patches".
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from app.models.dataset.spectralearth import SpectralEarthPatch


class PatchScreen(Protocol):
    """Decides, per pixel, whether a patch's spectrum is fit to use."""

    name: str

    def keep_mask(self, patch: SpectralEarthPatch) -> np.ndarray:
        """(rows, cols) bool: True where the pixel passes this screen."""
        ...


@dataclass
class NodataScreen:
    """Rejects pixels missing too much of their spectrum.

    `min_valid_band_fraction=0.6` matches the 40%-invalid-voxel rule in
    BandFilterConfig, so this screen drops exactly what that rule would, but early and
    counted. Pixels just under the limit keep their gaps for interpolation to fill.
    """

    min_valid_band_fraction: float = 0.6
    name: str = "nodata"

    def keep_mask(self, patch: SpectralEarthPatch) -> np.ndarray:
        return patch.validity().mean(axis=0) >= self.min_valid_band_fraction


@dataclass
class ScreenReport:
    keep: np.ndarray                                   # (rows, cols) bool, all screens ANDed
    rejected_by: dict[str, int] = field(default_factory=dict)   # screen -> pixels it rejected

    @property
    def keep_fraction(self) -> float:
        return float(self.keep.mean())

    def usable(self, min_keep_fraction: float = 0.5) -> bool:
        """Whether the patch is worth a training sample. 0.5, strictly exceeded, is the
        threshold the EnMAP intermediate sharder applies to its patches."""
        return self.keep_fraction > min_keep_fraction


class ScreenChain:
    """Runs screens in order; a pixel survives only if every screen keeps it. Each
    screen's count is of pixels it rejected that were still alive when it ran."""

    def __init__(self, screens: list[PatchScreen]):
        self.screens = screens

    def apply(self, patch: SpectralEarthPatch) -> ScreenReport:
        keep = np.ones(patch.raw.shape[1:], dtype=bool)
        rejected: dict[str, int] = {}
        for s in self.screens:
            mask = s.keep_mask(patch)
            # += so two screens sharing a name add up rather than overwrite.
            rejected[s.name] = rejected.get(s.name, 0) + int((keep & ~mask).sum())
            keep &= mask
        return ScreenReport(keep=keep, rejected_by=rejected)
