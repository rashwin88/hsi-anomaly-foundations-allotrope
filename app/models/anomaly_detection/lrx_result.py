"""
Result container for the Local RX anomaly detector.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class LocalRXResult:
    """Immutable result from a Local RX detection run."""

    lrx_score_map: np.ndarray          # (H, W) float, NaN where invalid or no bg
    spatial_mask: np.ndarray            # (H, W) bool — pixels eligible for scoring
    computed_mask: np.ndarray           # (H, W) bool — pixels that actually got a score
    good_band_indices: List[int]
    good_band_wavelengths: List[float]
    n_valid_pixels: int                 # pixels in spatial_mask
    n_scored_pixels: int                # pixels with a non-NaN score
    n_good_bands: int
    outer_window: int                   # background window half-size (pixels)
    inner_window: int                   # guard window half-size (pixels)

    def visualize(self) -> plt.Figure:
        """2-panel figure: spatial mask and LRX scores."""
        fig, (ax_mask, ax_lrx) = plt.subplots(1, 2, figsize=(14, 6))

        mask_rgb = np.zeros((*self.spatial_mask.shape, 3))
        mask_rgb[self.spatial_mask] = [0.2, 0.8, 0.3]
        mask_rgb[~self.spatial_mask] = [0.15, 0.15, 0.15]
        ax_mask.imshow(mask_rgb)
        ax_mask.set_title("Spatial Validity Mask")
        ax_mask.axis("off")

        cmap = plt.cm.inferno.copy()
        cmap.set_bad("lightgray")

        valid_scores = self.lrx_score_map[self.computed_mask]
        vmin = float(np.percentile(valid_scores, 2))
        vmax = float(np.percentile(valid_scores, 98))

        im = ax_lrx.imshow(
            self.lrx_score_map, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax_lrx.set_title("LRX Anomaly Scores (2nd–98th percentile)")
        ax_lrx.axis("off")
        fig.colorbar(im, ax=ax_lrx, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Local RX — {self.n_scored_pixels:,} scored pixels, "
            f"{self.n_good_bands} bands | "
            f"window outer={self.outer_window} inner={self.inner_window}",
            fontsize=11,
        )
        fig.tight_layout()
        return fig
