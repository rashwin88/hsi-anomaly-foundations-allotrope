"""
lrx-had-exp1: Destripe → LocalRX on a single PRISMA scene.

Pipeline: HE5 → PrismaDatasetBuilder → CombinedDestriper → LocalRXDetector
Scene: PRS_L2D_STD_20231229050902_20231229050907_0001

Parameters:
  outer_window = 25  (background annulus half-size)
  inner_window = 5   (guard half-size)
  stride       = 2   (subsample 4×, bilinear-interpolate back)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lrx-had-exp1")

HE5 = (
    PROJECT_ROOT
    / "tests/test_payloads/phase_2/Set-4/Hyper"
    / "PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
)
OUTPUT_DIR = Path(__file__).resolve().parent
N_ANOMALIES = 1000
OUTER_WINDOW = 25
INNER_WINDOW = 5
STRIDE       = 2


def main():
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
    from app.utils.data_transformations.composite_destriper import CombinedDestriper
    from app.detectors.local_rx_detector import LocalRXDetector

    # ── 1. Build vendable ─────────────────────────────────────────────
    logger.info("Step 1/4 — Building vendable from HE5")
    t0 = time.time()
    vendable = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(HE5))
    ).vend_dataset()
    cube                  = vendable.normalized_hyperspectral_cube
    validity              = vendable.validity_cube
    wavelengths           = vendable.band_cw_order
    spectral_family_order = vendable.spectral_family_order
    logger.info(
        "Vendable built in %.1fs — cube shape: %s",
        time.time() - t0, cube.shape,
    )

    # ── 2. Destripe ───────────────────────────────────────────────────
    logger.info("Step 2/4 — Destriping (FFT notch + moment-matching)")
    t1 = time.time()
    destriper = CombinedDestriper()
    destriped = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )
    logger.info("Destriping done in %.1fs | angles: %s", time.time() - t1, destriper.detected_angles)

    # ── 3. Fit + detect ───────────────────────────────────────────────
    logger.info(
        "Step 3/4 — LocalRX fit + detect "
        "(outer=%d inner=%d stride=%d)",
        OUTER_WINDOW, INNER_WINDOW, STRIDE,
    )
    t2 = time.time()
    detector = LocalRXDetector(vendable)
    detector.fit(band_failure_threshold=0.05)

    score_map = detector.detect(
        destriped,
        validity,
        outer_window=OUTER_WINDOW,
        inner_window=INNER_WINDOW,
        stride=STRIDE,
    )
    t_detect = time.time() - t2
    result = detector.result

    valid_scores = score_map[result.computed_mask]
    logger.info(
        "Detection done in %.1fs | scored=%d pixels | "
        "p2=%.2f median=%.2f p98=%.2f p99.9=%.2f max=%.2f",
        t_detect,
        result.n_scored_pixels,
        np.percentile(valid_scores, 2),
        np.median(valid_scores),
        np.percentile(valid_scores, 98),
        np.percentile(valid_scores, 99.9),
        valid_scores.max(),
    )

    # ── 4. Save figures ───────────────────────────────────────────────
    logger.info("Step 4/4 — Saving figures")

    # Standard LRX visualisation
    fig_lrx = result.visualize()
    out_lrx = OUTPUT_DIR / "lrx_scores.png"
    fig_lrx.savefig(str(out_lrx), dpi=150, bbox_inches="tight")
    plt.close(fig_lrx)
    logger.info("Saved: %s", out_lrx.name)

    # Top-N anomaly map (3-panel matching exp6 style)
    k = min(N_ANOMALIES, result.n_scored_pixels)
    threshold = float(np.partition(valid_scores, -k)[-k])
    anomaly_mask = (score_map >= threshold) & result.computed_mask
    if anomaly_mask.sum() > N_ANOMALIES:
        rows, cols  = np.where(anomaly_mask)
        order       = np.argsort(score_map[rows, cols])[::-1][:N_ANOMALIES]
        anomaly_mask = np.zeros(score_map.shape, dtype=bool)
        anomaly_mask[rows[order], cols[order]] = True

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    scene_id = HE5.stem
    fig.suptitle(
        f"{scene_id[:55]}\n"
        f"LRX | outer={OUTER_WINDOW} inner={INNER_WINDOW} stride={STRIDE} | "
        f"{result.n_scored_pixels:,} scored | top-{N_ANOMALIES} threshold: {threshold:.1f}",
        fontsize=9,
    )

    axes[0].imshow(result.spatial_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("Spatial Validity Mask")
    axes[0].axis("off")

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("lightgray")
    lo, hi = np.nanpercentile(score_map[result.computed_mask], [2, 98])
    im = axes[1].imshow(score_map, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    axes[1].set_title("LRX Anomaly Scores (p2–p98)")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    det_img = np.full(score_map.shape + (3,), 0.15)
    det_img[result.computed_mask] = [1.0, 1.0, 1.0]
    det_img[anomaly_mask]         = [1.0, 0.0, 0.0]
    axes[2].imshow(det_img, interpolation="nearest")
    axes[2].set_title(f"Detected Anomalies (top {N_ANOMALIES})")
    axes[2].axis("off")

    fig.tight_layout()
    out_det = OUTPUT_DIR / "lrx_detections.png"
    fig.savefig(str(out_det), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_det.name)

    logger.info(
        "Total time: %.1fs | Top-%d threshold=%.1f | anomalies=%d",
        time.time() - t0, N_ANOMALIES, threshold, int(anomaly_mask.sum()),
    )


if __name__ == "__main__":
    main()
