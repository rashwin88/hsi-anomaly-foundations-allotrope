"""
mnf-enmap-components: Visualize individual MNF components for each EnMAP scene.

For each scene, computes the MNF transform and plots a mosaic grid where
each tile is one MNF component mapped back to spatial (H, W).  This helps
identify which components carry signal vs. striping/noise.
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.detectors.mnf_compression_detector import MNFCompressionDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mnf-enmap-components")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
ENMAP_PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads"

# Show enough components to see where signal ends and noise/striping begins
N_COMPONENTS = 20
BAND_FAILURE_THRESHOLD = 0.05
MIN_BAND_COVERAGE = 0.95


def discover_enmap_folders() -> list[Path]:
    folders = sorted(
        p for p in ENMAP_PAYLOADS.iterdir()
        if p.is_dir() and p.name.startswith("ENMAP")
    )
    logger.info("Found %d EnMAP scene folders.", len(folders))
    return folders


def plot_mnf_mosaic(
    mnf_images: list[np.ndarray],
    eigenvalues: list[float],
    spatial_mask: np.ndarray,
    scene_id: str,
    out_path: Path,
) -> None:
    """Plot a mosaic grid of MNF component images."""
    n = len(mnf_images)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        if idx < n:
            img = mnf_images[idx]
            # Percentile stretch for display
            valid_vals = img[spatial_mask & ~np.isnan(img)]
            if len(valid_vals) > 0:
                vmin = np.percentile(valid_vals, 2)
                vmax = np.percentile(valid_vals, 98)
            else:
                vmin, vmax = 0, 1

            cmap = plt.cm.RdBu_r.copy()
            cmap.set_bad("black")
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            eig_str = f"{eigenvalues[idx]:.1f}" if idx < len(eigenvalues) else "?"
            ax.set_title(f"MNF {idx + 1}\n(λ={eig_str})", fontsize=9)
        else:
            ax.axis("off")
            continue

        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"{scene_id[:60]}\nMNF Components 1–{n} (sorted by descending SNR eigenvalue)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single(enmap_folder: Path) -> dict:
    scene_id = enmap_folder.name
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    t0 = time.time()

    # Build vendable
    vendable = EnmapDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(enmap_folder))
    ).vend_dataset()

    cube = vendable.normalized_hyperspectral_cube
    validity = vendable.validity_cube
    _, H, W = cube.shape

    # Fit MNF detector to get the transform
    detector = MNFCompressionDetector(vendable)
    detector.fit(
        n_components=N_COMPONENTS,
        band_failure_threshold=BAND_FAILURE_THRESHOLD,
        min_band_coverage=MIN_BAND_COVERAGE,
    )

    good = detector._good_indices
    mask = detector._spatial_mask
    n_components = detector._n_components

    # Extract good bands + band-mean fill (same logic as detector)
    working = cube[good].copy()
    for b_idx, b in enumerate(good):
        band_valid = validity[b].astype(bool)
        needs_fill = mask & (~band_valid)
        n_fill = int(needs_fill.sum())
        if n_fill > 0:
            donor = band_valid & mask
            fill_value = float(working[b_idx][donor].mean())
            working[b_idx][needs_fill] = fill_value

    pixels = working[:, mask].T  # (N, B_good)

    # Project into MNF space
    pixels_centered = pixels - detector._mnf_mean
    mnf_pixels = pixels_centered @ detector._mnf_components.T  # (N, n_comp)

    # Map each component back to (H, W)
    mnf_images = []
    for c in range(n_components):
        img = np.full((H, W), np.nan, dtype=np.float64)
        img[mask] = mnf_pixels[:, c]
        mnf_images.append(img)

    eigenvalues = detector._eigenvalues[:n_components].tolist()

    # Plot mosaic
    out_path = OUTPUT_DIR / f"{scene_id}_mnf_mosaic.png"
    plot_mnf_mosaic(mnf_images, eigenvalues, mask, scene_id, out_path)
    logger.info("Saved: %s", out_path.name)

    elapsed = time.time() - t0
    logger.info("Scene done in %.1fs", elapsed)

    return {
        "scene_id": scene_id,
        "n_good_bands": len(good),
        "n_components": n_components,
        "eigenvalues": eigenvalues,
        "fig_name": f"{scene_id}_mnf_mosaic.png",
        "time_s": round(elapsed, 1),
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# mnf-enmap-components: MNF Component Mosaics for EnMAP",
        "",
        "## Experiment",
        "",
        f"- **MNF components plotted:** {N_COMPONENTS}",
        f"- **Band failure threshold:** {BAND_FAILURE_THRESHOLD * 100:.0f}%",
        "- **Colormap:** RdBu_r (diverging), 2nd–98th percentile stretch",
        "- **Purpose:** Identify which MNF components carry signal vs. striping/noise",
        f"- **Scenes processed:** {len(results)}",
        "",
        "## Eigenvalue Spectra",
        "",
    ]

    for r in results:
        lines.append(f"### {r['scene_id'][:60]}")
        lines.append(f"- Bands: {r['n_good_bands']}, Components: {r['n_components']}")
        eig_str = ", ".join(f"{e:.1f}" for e in r["eigenvalues"])
        lines.append(f"- Eigenvalues: [{eig_str}]")
        lines.append("")

    lines += ["## Mosaics", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:60]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['fig_name']})")
        lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", summary_path)


def main():
    folders = discover_enmap_folders()
    if not folders:
        logger.error("No EnMAP scene folders found.")
        return

    results = []
    for folder in folders:
        try:
            results.append(run_single(folder))
        except Exception as e:
            logger.error("FAILED on %s: %s", folder.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes processed.", len(results), len(folders))


if __name__ == "__main__":
    main()
