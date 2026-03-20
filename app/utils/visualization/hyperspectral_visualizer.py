"""
Band mosaic visualizer for hyperspectral cubes.

Produces a single image showing every band as a small thumbnail arranged
in a grid, with wavelength labels.  Gives a quick overview of the entire
spectral range: noisy bands, atmospheric absorption, surface features.

Usage::

    from app.utils.visualization.hyperspectral_visualizer import plot_band_mosaic

    vendable = EnmapDatasetBuilder(...).vend_dataset()
    fig = plot_band_mosaic(vendable, title="My Scene", save_path="mosaic.png")

    # Or with good-band highlighting from a fitted detector:
    fig = plot_band_mosaic(
        vendable,
        good_band_indices=detector._good_indices,
        save_path="mosaic_filtered.png",
    )
"""

import logging
import math
from pathlib import Path
from typing import List, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from scipy.ndimage import zoom

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

logger = logging.getLogger(__name__)


def plot_band_mosaic(
    vendable,
    good_band_indices: List[int] | None = None,
    cmap: str = "inferno",
    percentile_clip: tuple[float, float] = (2, 98),
    max_output_pixels: int = 4096,
    title: str | None = None,
    save_path: str | Path | None = None,
    show_excluded: bool = True,
    family_boundary_nm: float = 1000.0,
) -> Figure:
    """
    Render every band of a hyperspectral cube as a thumbnail in a grid.

    Args:
        vendable:           VendableHyperspectralDataset or
                            VendableEnmapHyperspectralDataset.
        good_band_indices:  If provided, excluded bands are dimmed / red-labeled.
        cmap:               Matplotlib colormap name. Default ``'inferno'``.
        percentile_clip:    Per-band stretch percentiles. Default ``(2, 98)``.
        max_output_pixels:  Longest edge of the final mosaic. Default 4096.
        title:              Optional title shown in the top bar.
        save_path:          If given, save the figure as PNG.
        show_excluded:      Dim excluded bands and colour their labels red.
        family_boundary_nm: Wavelength (nm) used to detect the VNIR/SWIR
                            boundary when spectral_family_order is absent.

    Returns:
        matplotlib Figure.
    """
    cube = vendable.normalized_hyperspectral_cube        # (B, H, W)
    wavelengths = vendable.band_cw_order                 # list[float]
    validity = vendable.validity_cube                    # (B, H, W)
    spectral_families: list | None = getattr(vendable, "spectral_family_order", None)

    B, H, W = cube.shape
    aspect_ratio = W / H
    p_lo, p_hi = percentile_clip

    good_set = set(good_band_indices) if good_band_indices is not None else None

    n_cols = math.ceil(math.sqrt(B * aspect_ratio))
    n_rows = math.ceil(B / n_cols)
    thumb_size = max_output_pixels // max(n_rows, n_cols)
    thumb_size = max(thumb_size, 16)  # safety floor

    # Maintain band aspect ratio within each cell
    if H >= W:
        thumb_h = thumb_size
        thumb_w = max(1, round(thumb_size * W / H))
    else:
        thumb_w = thumb_size
        thumb_h = max(1, round(thumb_size * H / W))

    cell_h = thumb_h
    cell_w = thumb_w

    logger.info(
        "Band mosaic: %d bands -> %d x %d grid, thumb %d x %d, output ~%d x %d",
        B, n_rows, n_cols, thumb_h, thumb_w,
        n_rows * cell_h, n_cols * cell_w,
    )

    # ---- colormap setup ---------------------------------------------------
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad(color=(0.15, 0.15, 0.15, 1.0))     # dark gray for NaN

    # ---- find VNIR/SWIR boundary index ------------------------------------
    boundary_idx = _find_family_boundary(
        wavelengths, spectral_families, family_boundary_nm,
    )

    # ---- render thumbnails ------------------------------------------------
    # RGBA mosaic buffer
    bg = np.full((n_rows * cell_h, n_cols * cell_w, 4), 25, dtype=np.uint8)
    bg[:, :, 3] = 255  # opaque background

    font_size = max(6, thumb_size // 20)
    stroke = [pe.withStroke(linewidth=1.5, foreground="black")]

    fig, ax = plt.subplots(
        figsize=(n_cols * cell_w / 100, n_rows * cell_h / 100),
        dpi=100,
    )
    ax.imshow(bg)

    for idx in range(B):
        row = idx // n_cols
        col = idx % n_cols
        y0 = row * cell_h
        x0 = col * cell_w

        is_good = good_set is None or idx in good_set

        # --- per-band normalize ---
        band = cube[idx]
        valid = validity[idx].astype(bool)
        valid_vals = band[valid]

        if valid_vals.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin = float(np.percentile(valid_vals, p_lo))
            vmax = float(np.percentile(valid_vals, p_hi))

        normed = np.clip((band - vmin) / (vmax - vmin + 1e-10), 0, 1)
        normed[~valid] = np.nan

        # Downsample via scipy zoom (avoids skimage dependency)
        zoom_h = thumb_h / normed.shape[0]
        zoom_w = thumb_w / normed.shape[1]
        # Replace NaN before zoom (interpolation), restore after
        nan_mask = np.isnan(normed)
        normed_filled = normed.copy()
        normed_filled[nan_mask] = 0.0
        thumb = zoom(normed_filled, (zoom_h, zoom_w), order=1)
        thumb_nan = zoom(nan_mask.astype(float), (zoom_h, zoom_w), order=0) > 0.5
        thumb[thumb_nan] = np.nan

        # Map through colormap -> RGBA uint8
        rgba = (colormap(thumb) * 255).astype(np.uint8)

        # Dim excluded bands
        if show_excluded and not is_good:
            rgba[:, :, 3] = (rgba[:, :, 3] * 0.3).astype(np.uint8)

        # Place into canvas
        ax.imshow(
            rgba,
            extent=(x0, x0 + thumb_w, y0 + thumb_h, y0),
            interpolation="nearest",
            aspect="auto",
        )

        wl_text = f"{int(round(wavelengths[idx]))} nm"
        label_color = "white" if is_good else "red"
        ax.text(
            x0 + 2, y0 + thumb_h - 2,
            wl_text,
            fontsize=font_size,
            color=label_color,
            va="bottom",
            ha="left",
            path_effects=stroke,
        )

        if show_excluded and not is_good:
            rect = plt.Rectangle(
                (x0, y0), thumb_w, thumb_h,
                linewidth=1.2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)

    if boundary_idx is not None and 0 < boundary_idx < B:
        b_row = boundary_idx // n_cols
        b_col = boundary_idx % n_cols
        if b_col == 0:
            # Boundary falls at start of a row — draw horizontal line
            sep_y = b_row * cell_h
            ax.axhline(y=sep_y, color="cyan", linewidth=2, linestyle="-")
        else:
            # Draw vertical line within the row
            sep_x = b_col * cell_w
            y_top = b_row * cell_h
            y_bot = y_top + cell_h
            ax.plot([sep_x, sep_x], [y_top, y_bot], color="cyan",
                    linewidth=2, linestyle="-")

    # ---- axis limits & cleanup -------------------------------------------
    ax.set_xlim(0, n_cols * cell_w)
    ax.set_ylim(n_rows * cell_h, 0)
    ax.axis("off")

    # ---- title bar --------------------------------------------------------
    n_good = len(good_band_indices) if good_band_indices is not None else B
    sensor = _guess_sensor(wavelengths, B)
    title_parts = []
    if title:
        title_parts.append(title)
    title_parts.append(sensor)
    title_parts.append(f"{n_good}/{B} bands")
    title_parts.append(f"{cmap}")
    title_parts.append(f"per-band {p_lo}\u2013{p_hi}th pctl stretch")

    fig.suptitle(
        "  |  ".join(title_parts),
        fontsize=max(8, font_size + 2),
        color="white",
        fontweight="bold",
    )
    fig.patch.set_facecolor("#1a1a1a")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # ---- save -------------------------------------------------------------
    if save_path is not None:
        fig.savefig(
            str(save_path), dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
        logger.info("Band mosaic saved to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _find_family_boundary(
    wavelengths: Sequence[float],
    spectral_families: list | None,
    boundary_nm: float,
) -> int | None:
    """Return the index of the first SWIR band (or first band > boundary_nm)."""
    if spectral_families is not None:
        for i, fam in enumerate(spectral_families):
            if fam == SpectralFamily.SWIR:
                return i
        return None

    # Fallback: wavelength threshold
    for i, wl in enumerate(wavelengths):
        if wl > boundary_nm:
            return i
    return None


def _guess_sensor(wavelengths: Sequence[float], n_bands: int) -> str:
    """Best-effort sensor name from band count and wavelength range."""
    if n_bands >= 220:
        return "EnMAP"
    wl_max = max(wavelengths) if wavelengths else 0
    if n_bands > 60 and wl_max > 2000:
        return "PRISMA"
    if n_bands <= 10:
        return "Landsat 9"
    return f"{n_bands}-band HSI"
