"""
Result container for the MNF Compression + Global RX anomaly detector.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class MNFCompressionRXResult:
    """Immutable result from an MNF-compressed Global RX detection run."""

    rx_score_map: np.ndarray           # (H, W) float, NaN where invalid
    spatial_mask: np.ndarray            # (H, W) bool, True = valid pixel
    good_band_indices: List[int]
    good_band_wavelengths: List[float]
    n_valid_pixels: int
    n_good_bands: int
    n_components: int
    mnf_eigenvalues: List[float]

    def visualize(self) -> plt.Figure:
        """3-panel figure: spatial mask, RX scores, and MNF eigenvalue spectrum."""
        fig, (ax_mask, ax_rx, ax_eig) = plt.subplots(1, 3, figsize=(20, 6))

        # Validity mask
        mask_rgb = np.zeros((*self.spatial_mask.shape, 3))
        mask_rgb[self.spatial_mask] = [0.2, 0.8, 0.3]
        mask_rgb[~self.spatial_mask] = [0.15, 0.15, 0.15]
        ax_mask.imshow(mask_rgb)
        ax_mask.set_title("Spatial Validity Mask")
        ax_mask.axis("off")

        # RX scores
        display_map = self.rx_score_map.copy()
        valid_scores = display_map[~np.isnan(display_map)]
        vmin = np.percentile(valid_scores, 2)
        vmax = np.percentile(valid_scores, 98)
        display_map[np.isnan(display_map)] = vmin

        im = ax_rx.imshow(display_map, cmap="inferno", vmin=vmin, vmax=vmax)
        ax_rx.set_title("MNF-RX Anomaly Scores (2nd–98th percentile)")
        ax_rx.axis("off")
        fig.colorbar(im, ax=ax_rx, fraction=0.046, pad=0.04)

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
            f"MNF-GRX — {self.n_valid_pixels:,} valid pixels, "
            f"{self.n_good_bands} bands → {self.n_components} MNF components",
            fontsize=13,
        )
        fig.tight_layout()
        return fig
