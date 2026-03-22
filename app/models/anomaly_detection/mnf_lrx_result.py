"""
Result container for the MNF Compression + Local RX anomaly detector.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MNFCompressionLRXResult:
    """Immutable result from an MNF-compressed Local RX detection run."""

    lrx_score_map: np.ndarray          # (H, W) float, NaN where invalid or no bg
    spatial_mask: np.ndarray            # (H, W) bool — pixels eligible for scoring
    computed_mask: np.ndarray           # (H, W) bool — pixels that actually got a score
    good_band_indices: List[int]
    good_band_wavelengths: List[float]
    n_valid_pixels: int
    n_scored_pixels: int
    n_good_bands: int
    n_components: int
    mnf_eigenvalues: List[float]
    outer_window: int
    inner_window: int

    def visualize(self) -> plt.Figure:
        """3-panel figure: spatial mask, LRX scores, and MNF eigenvalue spectrum."""
        fig, (ax_mask, ax_lrx, ax_eig) = plt.subplots(1, 3, figsize=(20, 6))

        # Validity mask
        mask_rgb = np.zeros((*self.spatial_mask.shape, 3))
        mask_rgb[self.spatial_mask] = [0.2, 0.8, 0.3]
        mask_rgb[~self.spatial_mask] = [0.15, 0.15, 0.15]
        ax_mask.imshow(mask_rgb)
        ax_mask.set_title("Spatial Validity Mask")
        ax_mask.axis("off")

        # LRX scores
        cmap = plt.cm.inferno.copy()
        cmap.set_bad("lightgray")

        valid_scores = self.lrx_score_map[self.computed_mask]
        if len(valid_scores) > 0:
            vmin = float(np.percentile(valid_scores, 2))
            vmax = float(np.percentile(valid_scores, 98))
        else:
            vmin, vmax = 0.0, 1.0

        im = ax_lrx.imshow(
            self.lrx_score_map, cmap=cmap, vmin=vmin, vmax=vmax
        )
        ax_lrx.set_title("MNF-LRX Anomaly Scores (2nd–98th percentile)")
        ax_lrx.axis("off")
        fig.colorbar(im, ax=ax_lrx, fraction=0.046, pad=0.04)

        # Eigenvalue spectrum
        if self.mnf_eigenvalues:
            n_eig = len(self.mnf_eigenvalues)
            ax_eig.bar(
                range(n_eig), self.mnf_eigenvalues,
                color=["steelblue"] * self.n_components
                + ["lightgray"] * (n_eig - self.n_components),
            )
            ax_eig.axvline(
                self.n_components - 0.5, color="red", linestyle="--",
                label=f"cutoff ({self.n_components})",
            )
            ax_eig.set_xlabel("MNF Component")
            ax_eig.set_ylabel("Eigenvalue (SNR)")
            ax_eig.set_title("MNF Eigenvalue Spectrum")
            ax_eig.legend()

        fig.suptitle(
            f"MNF-LRX — {self.n_scored_pixels:,} scored / "
            f"{self.n_valid_pixels:,} valid pixels, "
            f"{self.n_good_bands} bands → {self.n_components} MNF components | "
            f"window outer={self.outer_window} inner={self.inner_window}",
            fontsize=11,
        )
        fig.tight_layout()
        return fig
