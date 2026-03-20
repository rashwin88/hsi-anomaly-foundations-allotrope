"""
rx-had-exp7: Compare Global RX anomaly score distributions with vs without destriping.

Per scene:
  - Run Global RX on raw cube and destriped cube (same band selection for both)
  - Overlay histogram of anomaly scores (raw vs destriped)
  - Collect percentile and summary statistics

Final output:
  - Per-scene histogram PNGs
  - summary.md with comparison table across all scenes
"""

import glob
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
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.data_transformations.composite_destriper import CombinedDestriper
from app.detectors.global_rx_detector import GlobalRXDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rx-had-exp7")
logging.getLogger("rx-had-exp7").setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
FAIL_THRESH = 0.05


def discover_he5_files() -> list[Path]:
    pattern = str(PROJECT_ROOT / "tests" / "test_payloads" / "**" / "*.he5")
    all_paths = [Path(p) for p in glob.glob(pattern, recursive=True)]
    seen: dict[str, Path] = {}
    for p in all_paths:
        if p.name not in seen:
            seen[p.name] = p
    deduped = list(seen.values())
    logger.info("Found %d HE5 files (%d unique)", len(all_paths), len(deduped))
    return deduped


def save_histogram(
    scene_id: str,
    raw_scores: np.ndarray,
    destriped_scores: np.ndarray,
) -> Path:
    """Overlay histogram: raw vs destriped anomaly scores."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f"{scene_id[:60]}", fontsize=10)

    # Panel 1 — full distribution (log-scale y)
    lo = min(raw_scores.min(), destriped_scores.min())
    hi = np.percentile(np.concatenate([raw_scores, destriped_scores]), 99.5)
    bins = np.linspace(lo, hi, 120)

    axes[0].hist(raw_scores, bins=bins, alpha=0.55, label="Raw", color="steelblue", density=True)
    axes[0].hist(destriped_scores, bins=bins, alpha=0.55, label="Destriped", color="coral", density=True)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("RX Score")
    axes[0].set_ylabel("Density (log)")
    axes[0].set_title("Full Distribution")
    axes[0].legend()

    # Panel 2 — upper tail (>= p90)
    p90_raw = np.percentile(raw_scores, 90)
    p90_des = np.percentile(destriped_scores, 90)
    tail_lo = min(p90_raw, p90_des)
    tail_hi = max(raw_scores.max(), destriped_scores.max())
    tail_bins = np.linspace(tail_lo, tail_hi, 80)

    axes[1].hist(raw_scores, bins=tail_bins, alpha=0.55, label="Raw", color="steelblue", density=True)
    axes[1].hist(destriped_scores, bins=tail_bins, alpha=0.55, label="Destriped", color="coral", density=True)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("RX Score")
    axes[1].set_ylabel("Density (log)")
    axes[1].set_title("Upper Tail (≥ p90)")
    axes[1].legend()

    fig.tight_layout()
    out = OUTPUT_DIR / f"{scene_id}_histogram.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def save_anomaly_maps(
    scene_id: str,
    spatial_mask: np.ndarray,
    raw_score_map: np.ndarray,
    des_score_map: np.ndarray,
    n_valid: int,
    n_good_bands: int,
) -> Path:
    """3-panel: validity mask | raw RX scores | destriped RX scores."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"{scene_id[:55]}\n"
        f"{n_valid:,} valid pixels · {n_good_bands} bands",
        fontsize=9,
    )

    # Panel 1 — spatial validity mask
    axes[0].imshow(spatial_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Spatial Validity Mask")
    axes[0].axis("off")

    # Shared colour scale from both score maps (p2–p99.5 of combined valid scores)
    raw_valid = raw_score_map[spatial_mask]
    des_valid = des_score_map[spatial_mask]
    combined = np.concatenate([raw_valid, des_valid])
    lo, hi = np.percentile(combined, [2, 99.5])

    # Hot colormap: background is dark, outliers saturate to bright white/yellow
    cmap = plt.cm.hot.copy()
    cmap.set_bad("lightgray")
    cmap.set_over("cyan")  # scores above p99.5 clip to cyan for visibility

    # Panel 2 — raw RX scores
    im_raw = axes[1].imshow(
        np.where(spatial_mask, raw_score_map, np.nan),
        cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest",
    )
    axes[1].set_title("RX Scores — Raw")
    axes[1].axis("off")
    fig.colorbar(im_raw, ax=axes[1], fraction=0.046, pad=0.04, extend="max")

    # Panel 3 — destriped RX scores
    im_des = axes[2].imshow(
        np.where(spatial_mask, des_score_map, np.nan),
        cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest",
    )
    axes[2].set_title("RX Scores — Destriped")
    axes[2].axis("off")
    fig.colorbar(im_des, ax=axes[2], fraction=0.046, pad=0.04, extend="max")

    fig.tight_layout()
    out = OUTPUT_DIR / f"{scene_id}_anomaly_maps.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run_single(he5_path: Path) -> dict:
    scene_id = he5_path.stem
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    t0 = time.time()

    vendable = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    ).vend_dataset()

    cube                  = vendable.normalized_hyperspectral_cube
    validity              = vendable.validity_cube
    wavelengths           = vendable.band_cw_order
    spectral_family_order = vendable.spectral_family_order

    # Fit detector once (band selection is the same for raw and destriped)
    detector = GlobalRXDetector(vendable)
    detector.fit(band_failure_threshold=FAIL_THRESH)

    # --- Raw RX ---
    t_raw = time.time()
    raw_score_map = detector.detect(cube, validity)
    raw_result = detector.result
    t_raw = time.time() - t_raw

    spatial_mask = raw_result.spatial_mask
    raw_valid = raw_score_map[spatial_mask]

    # --- Destriped RX ---
    t_des = time.time()
    destriper = CombinedDestriper()
    destriped_cube = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )
    des_score_map = detector.detect(destriped_cube, validity)
    des_result = detector.result
    t_des = time.time() - t_des

    des_valid = des_score_map[spatial_mask]

    # Histogram
    fig_path = save_histogram(scene_id, raw_valid, des_valid)

    # 3-panel anomaly maps
    map_path = save_anomaly_maps(
        scene_id, spatial_mask, raw_score_map, des_score_map,
        n_valid=int(spatial_mask.sum()), n_good_bands=raw_result.n_good_bands,
    )

    elapsed = time.time() - t0
    logger.info(
        "Done in %.1fs | bands=%d | valid_px=%d | fig=%s",
        elapsed, raw_result.n_good_bands, int(spatial_mask.sum()), fig_path.name,
    )

    def pcts(arr):
        return {
            f"p{p}": round(float(np.percentile(arr, p)), 2)
            for p in [25, 50, 75, 90, 95, 99, 99.9]
        }

    return {
        "scene_id":       scene_id,
        "n_good_bands":   raw_result.n_good_bands,
        "n_valid":        int(spatial_mask.sum()),
        "raw_mean":       round(float(raw_valid.mean()), 2),
        "raw_std":        round(float(raw_valid.std()), 2),
        "raw_max":        round(float(raw_valid.max()), 2),
        "raw_pcts":       pcts(raw_valid),
        "des_mean":       round(float(des_valid.mean()), 2),
        "des_std":        round(float(des_valid.std()), 2),
        "des_max":        round(float(des_valid.max()), 2),
        "des_pcts":       pcts(des_valid),
        "time_raw_s":     round(t_raw, 1),
        "time_des_s":     round(t_des, 1),
        "fig_name":       fig_path.name,
        "map_name":       map_path.name,
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# rx-had-exp7: Raw vs Destriped Global RX Score Distributions",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → GlobalRXDetector (raw cube) **vs** "
        "CombinedDestriper → GlobalRXDetector (destriped cube)",
        "",
        "Band selection (fit) is shared — only the cube fed to `detect()` differs.",
        "",
        "## Comparison Table",
        "",
        "| Scene | Bands | Valid Px | "
        "Raw Mean | Raw Std | Raw p99 | Raw Max | "
        "Des Mean | Des Std | Des p99 | Des Max |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:40]} | {r['n_good_bands']} | {r['n_valid']:,} | "
            f"{r['raw_mean']} | {r['raw_std']} | {r['raw_pcts']['p99']} | {r['raw_max']} | "
            f"{r['des_mean']} | {r['des_std']} | {r['des_pcts']['p99']} | {r['des_max']} |"
        )

    # Detailed percentile table
    lines += [
        "",
        "## Percentile Breakdown",
        "",
        "| Scene | Mode | p25 | p50 | p75 | p90 | p95 | p99 | p99.9 | Max |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        short = r["scene_id"][:30]
        rp = r["raw_pcts"]
        dp = r["des_pcts"]
        lines.append(
            f"| {short} | Raw | {rp['p25']} | {rp['p50']} | {rp['p75']} | "
            f"{rp['p90']} | {rp['p95']} | {rp['p99']} | {rp['p99.9']} | {r['raw_max']} |"
        )
        lines.append(
            f"| | Destriped | {dp['p25']} | {dp['p50']} | {dp['p75']} | "
            f"{dp['p90']} | {dp['p95']} | {dp['p99']} | {dp['p99.9']} | {r['des_max']} |"
        )

    lines += ["", "## Anomaly Maps", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:50]}")
        lines.append("")
        lines.append(f"![{r['scene_id']} maps]({r['map_name']})")
        lines.append("")

    lines += ["", "## Histograms", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:50]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['fig_name']})")
        lines.append("")

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))
    logger.info("Summary written to summary.md")


def main():
    he5_files = discover_he5_files()
    if not he5_files:
        logger.error("No HE5 files found.")
        return

    results = []
    for path in he5_files:
        try:
            results.append(run_single(path))
        except Exception as e:
            logger.error("FAILED on %s: %s", path.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes.", len(results), len(he5_files))


if __name__ == "__main__":
    main()
