"""
Result container for the Thermal GRX anomaly detector.
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class ThermalGRXResult:
    """Immutable result from a Thermal Global RX detection run."""

    rx_score_map: np.ndarray    # (H, W) float, NaN where invalid
    spatial_mask: np.ndarray    # (H, W) bool, True = valid pixel
    n_valid_pixels: int

    def visualize(self) -> plt.Figure:
        """2-panel figure: spatial mask and RX scores."""
        fig, (ax_mask, ax_rx) = plt.subplots(1, 2, figsize=(14, 6))

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
        if len(valid_scores) > 0:
            vmin = np.percentile(valid_scores, 2)
            vmax = np.percentile(valid_scores, 98)
        else:
            vmin, vmax = 0.0, 1.0
        display_map[np.isnan(display_map)] = vmin

        im = ax_rx.imshow(display_map, cmap="inferno", vmin=vmin, vmax=vmax)
        ax_rx.set_title("Thermal GRX Anomaly Scores (2nd–98th percentile)")
        ax_rx.axis("off")
        fig.colorbar(im, ax=ax_rx, fraction=0.046, pad=0.04)

        fig.suptitle(
            f"Thermal GRX — {self.n_valid_pixels:,} valid pixels, "
            f"single-band thermal",
            fontsize=13,
        )
        fig.tight_layout()
        return fig
