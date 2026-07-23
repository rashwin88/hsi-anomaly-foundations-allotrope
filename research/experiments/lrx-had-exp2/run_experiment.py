"""
lrx-had-exp2: Destripe → LocalRX on all 5 PRISMA scenes, stride=4.

Pipeline: HE5 → PrismaDatasetBuilder → CombinedDestriper → LocalRXDetector
Output per scene: 3-panel PNG (validity mask | LRX scores | top-1000 detections)
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lrx-had-exp2")

OUTPUT_DIR   = Path(__file__).resolve().parent
N_ANOMALIES  = 1000
OUTER_WINDOW = 25
INNER_WINDOW = 5
STRIDE       = 4
FAIL_THRESH  = 0.05


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


def run_single(he5_path: Path) -> dict:
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
    from app.utils.data_transformations.composite_destriper import CombinedDestriper
    from app.detectors.local_rx_detector import LocalRXDetector

    scene_id = he5_path.stem
    logger.info("=" * 70)
    logger.info("Scene: %s", scene_id)

    t_start = time.time()

    # 1. Build vendable
    logger.info("[1/4] Building vendable")
    vendable = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    ).vend_dataset()
    cube                  = vendable.normalized_hyperspectral_cube
    validity              = vendable.validity_cube
    wavelengths           = vendable.band_cw_order
    spectral_family_order = vendable.spectral_family_order
    logger.info("[1/4] Done — cube %s", cube.shape)

    # 2. Destripe
    logger.info("[2/4] Destriping")
    t1 = time.time()
    destriper = CombinedDestriper()
    destriped = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )
    logger.info("[2/4] Done in %.1fs | angles: %s", time.time() - t1, destriper.detected_angles)

    # 3. LRX fit + detect
    logger.info(
        "[3/4] LocalRX fit + detect (outer=%d inner=%d stride=%d)",
        OUTER_WINDOW, INNER_WINDOW, STRIDE,
    )
    t2 = time.time()
    detector = LocalRXDetector(vendable)
    detector.fit(band_failure_threshold=FAIL_THRESH)
    score_map = detector.detect(
        destriped, validity,
        outer_window=OUTER_WINDOW,
        inner_window=INNER_WINDOW,
        stride=STRIDE,
    )
    t_detect = time.time() - t2
    result = detector.result

    valid_scores = score_map[result.computed_mask]
    percs = {
        "p2":    float(np.percentile(valid_scores, 2)),
        "p25":   float(np.percentile(valid_scores, 25)),
        "median":float(np.median(valid_scores)),
        "p75":   float(np.percentile(valid_scores, 75)),
        "p98":   float(np.percentile(valid_scores, 98)),
        "p99":   float(np.percentile(valid_scores, 99)),
        "p999":  float(np.percentile(valid_scores, 99.9)),
        "max":   float(valid_scores.max()),
    }
    logger.info(
        "[3/4] Done in %.1fs | scored=%d | p2=%.2f median=%.2f p98=%.2f p99.9=%.2f max=%.2f",
        t_detect, result.n_scored_pixels,
        percs["p2"], percs["median"], percs["p98"], percs["p999"], percs["max"],
    )

    # 4. Save figure
    logger.info("[4/4] Saving figure")
    k         = min(N_ANOMALIES, result.n_scored_pixels)
    threshold = float(np.partition(valid_scores, -k)[-k])
    anomaly_mask = (score_map >= threshold) & result.computed_mask
    if anomaly_mask.sum() > N_ANOMALIES:
        rows, cols   = np.where(anomaly_mask)
        order        = np.argsort(score_map[rows, cols])[::-1][:N_ANOMALIES]
        anomaly_mask = np.zeros(score_map.shape, dtype=bool)
        anomaly_mask[rows[order], cols[order]] = True

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
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
    fig_path = OUTPUT_DIR / f"{scene_id}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[4/4] Saved: %s", fig_path.name)

    t_total = time.time() - t_start
    logger.info("Scene total: %.1fs", t_total)

    return {
        "scene_id":        scene_id,
        "n_good_bands":    result.n_good_bands,
        "n_scored":        result.n_scored_pixels,
        "threshold":       round(threshold, 2),
        "percs":           percs,
        "t_detect_s":      round(t_detect, 1),
        "t_total_s":       round(t_total, 1),
        "fig_name":        fig_path.name,
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# lrx-had-exp2: Local RX on All 5 PRISMA Scenes (stride=4)",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → CombinedDestriper (20 probes) → LocalRXDetector",
        f"Parameters: outer_window={OUTER_WINDOW}, inner_window={INNER_WINDOW}, stride={STRIDE}",
        "",
        "| Scene | Bands | Scored Px | threshold | p2 | median | p98 | p99.9 | max | Detect (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        p = r["percs"]
        lines.append(
            f"| {r['scene_id'][:40]} | {r['n_good_bands']} | {r['n_scored']:,} | "
            f"{r['threshold']} | {p['p2']:.2f} | {p['median']:.2f} | "
            f"{p['p98']:.2f} | {p['p999']:.2f} | {p['max']:.1f} | {r['t_detect_s']} |"
        )
    lines += ["", "## Visualizations", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:55]}")
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
            logger.error("FAILED %s: %s", path.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes.", len(results), len(he5_files))


if __name__ == "__main__":
    main()
