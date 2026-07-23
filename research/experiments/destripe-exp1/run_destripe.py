"""
destripe-exp1: Run CompositeDestriper (FFT notch + tilted moment-matching)
on two PRISMA scenes with full diagnostics.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.data_transformations.composite_destriper import CombinedDestriper
from app.utils.data_transformations.frequency_domain_destriper import (
    FrequencyDestripeDiagnostics,
    PAD_WIDTH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("destripe-exp1")

OUTPUT_DIR = Path(__file__).resolve().parent

def discover_he5_files() -> list[str]:
    """Find all .he5 files under test_payloads, deduplicated by filename."""
    import glob
    pattern = str(PROJECT_ROOT / "tests" / "test_payloads" / "**" / "*.he5")
    all_paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    seen: dict[str, Path] = {}
    for p in all_paths:
        if p.name not in seen:
            seen[p.name] = p
    deduped = [str(p) for p in seen.values()]
    logger.info("Found %d HE5 files (%d unique)", len(all_paths), len(deduped))
    return deduped


def build_vendable(path: str):
    builder = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=path)
    )
    return builder.vend_dataset()


def plot_diagnostics(
    diag: FrequencyDestripeDiagnostics, scene_id: str, output_path: Path,
    validity_2d: np.ndarray = None,
    sigma_stages: tuple = None,
):
    """7-panel diagnostic figure (3x3 grid, last two empty)."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    # Panel 1: Power vs angle with threshold line
    ax = axes[0, 0]
    ax.plot(diag.angles, diag.power_sums, linewidth=0.8)
    ax.axhline(
        diag.significance_threshold,
        color="red", linestyle="--", label=f"1.5x median = {diag.significance_threshold:.0f}",
    )
    if diag.detected_angle is not None:
        ax.axvline(diag.detected_angle, color="orange", linestyle=":", label=f"Detected: {diag.detected_angle:.1f}°")
    ax.set_xlabel("Angle (degrees)")
    ax.set_ylabel("Sum of log power")
    ax.set_title("Radial Power vs Angle")
    ax.legend(fontsize=8)

    # Panel 2: Log power spectrum before, with detected angle line
    ax = axes[0, 1]
    im = ax.imshow(diag.log_power_before, cmap="viridis", aspect="auto")
    if diag.detected_angle is not None:
        H, W = diag.log_power_before.shape
        cy, cx = H // 2, W // 2
        angle_rad = np.deg2rad(diag.detected_angle)
        r = min(cy, cx) * 0.9
        ax.plot(
            [cx - r * np.cos(angle_rad), cx + r * np.cos(angle_rad)],
            [cy - r * np.sin(angle_rad), cy + r * np.sin(angle_rad)],
            color="red", linewidth=1, linestyle="--",
        )
    ax.set_title("Log Power Spectrum (before)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: Notch filter
    ax = axes[0, 2]
    if diag.notch_filter is not None:
        im = ax.imshow(diag.notch_filter, cmap="gray", vmin=0, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "No stripe detected", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Notch Filter")
    ax.axis("off")

    # Panel 4: Log power spectrum after
    ax = axes[1, 0]
    if diag.log_power_after is not None:
        im = ax.imshow(diag.log_power_after, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Log Power Spectrum (after)")
    ax.axis("off")

    # Panel 5: Band before (spatial)
    ax = axes[1, 1]
    if diag.band_before is not None:
        valid = ~np.isnan(diag.band_before) & (diag.band_before != 0)
        vmin_b = np.percentile(diag.band_before[valid], 2) if valid.any() else 0
        vmax_b = np.percentile(diag.band_before[valid], 98) if valid.any() else 1
        ax.imshow(diag.band_before, cmap="gray", vmin=vmin_b, vmax=vmax_b)
    ax.set_title(f"Band {diag.representative_band_index} — Before")
    ax.axis("off")

    # Panel 6: Band after (spatial)
    ax = axes[1, 2]
    if diag.band_after is not None and diag.band_before is not None:
        ax.imshow(diag.band_after, cmap="gray", vmin=vmin_b, vmax=vmax_b)
    ax.set_title(f"Band {diag.representative_band_index} — After")
    ax.axis("off")

    # Panel 7: Column-mean profile before and after
    ax = axes[2, 0]
    if diag.band_before is not None and diag.band_after is not None:
        validity_band = validity_2d
        W = diag.band_before.shape[1]
        col_means_before = np.zeros(W)
        col_means_after = np.zeros(W)
        for j in range(W):
            col_valid = validity_band[:, j].astype(bool)
            if col_valid.sum() > 0:
                col_means_before[j] = diag.band_before[col_valid, j].mean()
                col_means_after[j] = diag.band_after[col_valid, j].mean()
            else:
                col_means_before[j] = np.nan
                col_means_after[j] = np.nan

        ax.plot(col_means_before, label="Before", alpha=0.7, linewidth=0.6)
        ax.plot(col_means_after, label="After", alpha=0.7, linewidth=0.6)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Mean reflectance (valid pixels)")
        ax.legend(fontsize=8)

        if sigma_stages and sigma_stages[0] is not None:
            ax.set_title(
                f"Column-Mean Profile (band {diag.representative_band_index})\n"
                f"σ original: {sigma_stages[0]:.6f} → FFT: {sigma_stages[1]:.6f} "
                f"→ combined: {sigma_stages[2]:.6f}"
            )
        else:
            std_before = np.nanstd(col_means_before)
            std_after = np.nanstd(col_means_after)
            ax.set_title(
                f"Column-Mean Profile (band {diag.representative_band_index})\n"
                f"σ before: {std_before:.6f} → σ after: {std_after:.6f}"
            )
    else:
        ax.text(0.5, 0.5, "No stripe detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Column-Mean Profile")

    # Hide unused panels
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")

    angle_str = f"{diag.detected_angle:.1f}°" if diag.detected_angle is not None else "None (below threshold)"
    fig.suptitle(
        f"Destripe Diagnostics — {scene_id}\n"
        f"Detected angle: {angle_str} | Pad: {PAD_WIDTH}px",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved diagnostics: %s", output_path.name)


def main():
    destriper = CombinedDestriper()

    scenes = discover_he5_files()
    if not scenes:
        logger.error("No HE5 files found.")
        return

    for scene_path in scenes:
        scene_id = Path(scene_path).stem
        logger.info("=" * 60)
        logger.info("Processing: %s", scene_id)

        vendable = build_vendable(scene_path)
        cube = vendable.normalized_hyperspectral_cube
        validity = vendable.validity_cube

        destriped = destriper.transform(
            cube,
            validity_mask=validity,
            diagnostics=True,
        )

        combined_diag = destriper.diagnostics
        fft_diag = combined_diag.fft_diagnostics
        if fft_diag is None:
            fft_diag = FrequencyDestripeDiagnostics()

        rep = fft_diag.representative_band_index

        # Override band_after with the final combined result
        if rep is not None:
            fft_diag.band_after = destriped[rep].copy()

        fig_path = OUTPUT_DIR / f"{scene_id}_diagnostics.png"
        rep_validity = validity[rep] if rep is not None else None
        plot_diagnostics(
            fft_diag, scene_id, fig_path,
            validity_2d=rep_validity,
            sigma_stages=(
                combined_diag.sigma_original,
                combined_diag.sigma_after_fft,
                combined_diag.sigma_after_combined,
            ),
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
