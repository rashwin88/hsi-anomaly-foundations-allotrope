"""
Result container for the StatisticalEnsembler (GRX + LRX).
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from app.models.anomaly_detection.rx_result import GlobalRXResult
from app.models.anomaly_detection.lrx_result import LocalRXResult


FUSION_STRATEGIES = ("product", "maximum", "mean")


def cdf_normalize(score_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Rank-transform valid scores to [0, 1] (CDF normalization).

    Invalid (NaN) pixels stay NaN. Rank ties receive the average rank.
    """
    out = np.full_like(score_map, np.nan)
    valid = mask & ~np.isnan(score_map)
    if valid.sum() == 0:
        return out
    ranks = rankdata(score_map[valid], method="average")
    out[valid] = (ranks - 1) / (ranks.size - 1) if ranks.size > 1 else 0.5
    return out


def fuse_maps(
    norm_grx: np.ndarray,
    norm_lrx: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Produce fused score maps for each strategy.

    Both inputs must be CDF-normalized [0, 1] with NaN outside `mask`.
    """
    fused = {}
    both_valid = mask & ~np.isnan(norm_grx) & ~np.isnan(norm_lrx)

    for strategy in FUSION_STRATEGIES:
        out = np.full_like(norm_grx, np.nan)
        if strategy == "product":
            out[both_valid] = norm_grx[both_valid] * norm_lrx[both_valid]
        elif strategy == "maximum":
            out[both_valid] = np.maximum(norm_grx[both_valid], norm_lrx[both_valid])
        elif strategy == "mean":
            out[both_valid] = (norm_grx[both_valid] + norm_lrx[both_valid]) / 2.0
        fused[strategy] = out

    return fused


@dataclass(frozen=True)
class EnsembleRXResult:
    """Immutable result from a GRX + LRX ensemble detection run."""

    grx_result: GlobalRXResult
    lrx_result: LocalRXResult
    normalized_grx_map: np.ndarray          # (H, W) CDF-normalized [0,1], NaN invalid
    normalized_lrx_map: np.ndarray          # (H, W) CDF-normalized [0,1], NaN invalid
    fused_score_maps: Dict[str, np.ndarray] # strategy -> (H, W) fused scores
    spatial_mask: np.ndarray                # (H, W) bool — intersection of GRX & LRX masks
    good_band_indices: List[int]
    good_band_wavelengths: List[float]
    n_valid_pixels: int
    n_good_bands: int
    destriped: bool

    def visualize(self, strategy: str = "product") -> plt.Figure:
        """4-panel figure: GRX, LRX, normalized pair, and fused map."""
        if strategy not in self.fused_score_maps:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Available: {list(self.fused_score_maps)}"
            )

        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        ax_grx, ax_lrx, ax_ngrx, ax_fuse = axes

        cmap = plt.cm.inferno.copy()
        cmap.set_bad("lightgray")

        def _show(ax, data, title):
            valid = data[~np.isnan(data)]
            if valid.size == 0:
                ax.set_title(title)
                ax.axis("off")
                return
            vmin = np.percentile(valid, 2)
            vmax = np.percentile(valid, 98)
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        _show(ax_grx,  self.grx_result.rx_score_map,   "GRX (raw)")
        _show(ax_lrx,  self.lrx_result.lrx_score_map,  "LRX (raw)")
        _show(ax_ngrx, self.normalized_grx_map,         "GRX (CDF-norm)")
        _show(ax_fuse, self.fused_score_maps[strategy], f"Fused ({strategy})")

        destrip_label = "destriped" if self.destriped else "raw"
        fig.suptitle(
            f"Statistical Ensemble — {self.n_valid_pixels:,} valid pixels, "
            f"{self.n_good_bands} bands ({destrip_label})",
            fontsize=13,
        )
        fig.tight_layout()
        return fig
