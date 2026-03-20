"""
rx-had-exp6: Destripe → GlobalRX → top-1000 anomaly detection map.

Same pipeline as exp5 (20 partitioned probes). Output per scene:
  3-panel PNG:
    1. Spatial validity mask (green/black)
    2. RX anomaly score map (inferno, p2–p98 clip)
    3. Detected anomalies — top 1000 scoring pixels in red, valid swath in white
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
logger = logging.getLogger("rx-had-exp6")
logging.getLogger("rx-had-exp6").setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
FAIL_THRESH = 0.05
N_ANOMALIES = 1000


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


def save_figure(
    scene_id: str,
    score_map: np.ndarray,
    spatial_mask: np.ndarray,
    anomaly_mask: np.ndarray,
    n_valid: int,
    n_good_bands: int,
    threshold_score: float,
) -> Path:
    """3-panel: validity mask | RX scores | top-N anomalies."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"{scene_id[:55]}\n"
        f"{n_valid:,} valid pixels · {n_good_bands} bands · "
        f"top-{N_ANOMALIES} threshold: {threshold_score:.1f}",
        fontsize=9,
    )

    # Panel 1 — spatial mask
    axes[0].imshow(spatial_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Spatial Validity Mask")
    axes[0].axis("off")

    # Panel 2 — RX scores
    cmap_score = plt.cm.inferno.copy()
    cmap_score.set_bad("lightgray")
    lo, hi = np.nanpercentile(score_map[spatial_mask], [2, 98])
    im = axes[1].imshow(
        np.where(spatial_mask, score_map, np.nan),
        cmap=cmap_score, vmin=lo, vmax=hi, interpolation="nearest",
    )
    axes[1].set_title("RX Anomaly Scores (p2–p98)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3 — detected anomalies
    # White = valid swath, red = top-N anomaly, gray = invalid
    detection_img = np.full(score_map.shape + (3,), fill_value=0.15)  # dark gray bg
    # Valid swath → white
    detection_img[spatial_mask] = [1.0, 1.0, 1.0]
    # Top-N anomalies → red
    detection_img[anomaly_mask] = [1.0, 0.0, 0.0]

    axes[2].imshow(detection_img, interpolation="nearest")
    axes[2].set_title(f"Detected Anomalies (top {N_ANOMALIES})")
    axes[2].axis("off")

    fig.tight_layout()
    out = OUTPUT_DIR / f"{scene_id}.png"
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

    # Destripe
    destriper = CombinedDestriper()
    destriped = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )

    # RX
    detector = GlobalRXDetector(vendable)
    detector.fit(band_failure_threshold=FAIL_THRESH)
    score_map = detector.detect(destriped, validity)
    result    = detector.result

    spatial_mask = result.spatial_mask
    valid_scores = score_map[spatial_mask]
    n_valid      = int(spatial_mask.sum())

    # Top-N anomaly mask
    # Find the N-th highest score threshold among valid pixels
    k = min(N_ANOMALIES, n_valid)
    threshold_score = float(np.partition(valid_scores, -k)[-k])
    anomaly_mask = (score_map >= threshold_score) & spatial_mask

    # If ties push us above exactly N, trim down to N by score rank
    if anomaly_mask.sum() > N_ANOMALIES:
        rows, cols = np.where(anomaly_mask)
        scores_at  = score_map[rows, cols]
        top_k      = np.argsort(scores_at)[::-1][:N_ANOMALIES]
        anomaly_mask = np.zeros_like(spatial_mask)
        anomaly_mask[rows[top_k], cols[top_k]] = True

    fig_path = save_figure(
        scene_id,
        score_map,
        spatial_mask,
        anomaly_mask,
        n_valid,
        result.n_good_bands,
        threshold_score,
    )
    logger.info(
        "Done in %.1fs | bands=%d | valid_px=%d | threshold=%.1f | fig=%s",
        time.time() - t0,
        result.n_good_bands,
        n_valid,
        threshold_score,
        fig_path.name,
    )

    valid_scores_all = score_map[~np.isnan(score_map)]
    return {
        "scene_id":        scene_id,
        "n_good_bands":    result.n_good_bands,
        "n_valid":         n_valid,
        "n_anomalies":     int(anomaly_mask.sum()),
        "threshold_score": round(threshold_score, 2),
        "score_p99":       round(float(np.percentile(valid_scores_all, 99)), 2),
        "score_max":       round(float(valid_scores_all.max()), 2),
        "fig_name":        fig_path.name,
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# rx-had-exp6: Top-1000 Anomaly Detection",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → CombinedDestriper (20 probes: 10 VNIR + 10 SWIR) "
        "→ GlobalRXDetector → top-1000 anomaly map",
        "",
        "| Scene | Bands | Valid Px | Threshold score | p99 | Max | Anomalies |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:40]} | {r['n_good_bands']} | {r['n_valid']:,} | "
            f"{r['threshold_score']} | {r['score_p99']} | {r['score_max']} | "
            f"{r['n_anomalies']} |"
        )
    lines += ["", "## Visualizations", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:50]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['fig_name']})")
        lines.append("")

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))
    logger.info("Summary written.")


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
