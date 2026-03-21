"""
mnf-enmap: Run MNFCompressionDetector on all EnMAP L2A scenes in test_payloads.

Applies MNF dimensionality reduction (n → n_components bands) before
computing Global RX scores.  Discovers EnMAP scene folders, builds
vendables via EnmapDatasetBuilder, runs detection, saves visualizations,
and writes a summary markdown.
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
logger = logging.getLogger("mnf-enmap")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
ENMAP_PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads"

# ---- Experiment parameters ----
N_COMPONENTS = 5
BAND_FAILURE_THRESHOLD = 0.05
MIN_BAND_COVERAGE = 0.95


def discover_enmap_folders() -> list[Path]:
    """Find all EnMAP L2A scene folders under the test payloads dir."""
    folders = sorted(
        p for p in ENMAP_PAYLOADS.iterdir()
        if p.is_dir() and p.name.startswith("ENMAP")
    )
    logger.info("Found %d EnMAP scene folders.", len(folders))
    return folders


def run_single(enmap_folder: Path) -> dict:
    """Build vendable, run MNF-RX, save visualization. Returns summary dict."""
    scene_id = enmap_folder.name
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(exist_ok=True)
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    # Build vendable
    t0 = time.time()
    vendable = EnmapDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(enmap_folder))
    ).vend_dataset()

    cube = vendable.normalized_hyperspectral_cube
    validity = vendable.validity_cube
    _, H, W = cube.shape
    t_build = time.time() - t0
    logger.info("Cube shape: %s (built in %.1fs)", cube.shape, t_build)

    # Run MNF compression + GRX
    t1 = time.time()
    detector = MNFCompressionDetector(vendable)
    detector.fit(
        n_components=N_COMPONENTS,
        band_failure_threshold=BAND_FAILURE_THRESHOLD,
        min_band_coverage=MIN_BAND_COVERAGE,
    )
    score_map = detector.detect(cube, validity)
    t_detect = time.time() - t1

    result = detector.result

    # Save 3-panel visualization (mask, scores, eigenvalue spectrum)
    fig = result.visualize()
    fig_path = scene_dir / "mnf_rx_panels.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path)

    valid_scores = score_map[result.spatial_mask & ~np.isnan(score_map)]

    elapsed = time.time() - t0
    logger.info("Scene done in %.1fs", elapsed)

    return {
        "scene_id": scene_id,
        "cube_shape": f"{cube.shape[0]}x{H}x{W}",
        "n_total_bands": cube.shape[0],
        "n_good_bands": result.n_good_bands,
        "n_components": result.n_components,
        "n_valid_pixels": result.n_valid_pixels,
        "n_total_pixels": H * W,
        "pixel_band_ratio": round(result.n_valid_pixels / result.n_good_bands, 1),
        "score_min": round(float(valid_scores.min()), 4),
        "score_max": round(float(valid_scores.max()), 4),
        "score_median": round(float(np.median(valid_scores)), 4),
        "score_p99": round(float(np.percentile(valid_scores, 99)), 4),
        "build_time_s": round(t_build, 2),
        "detect_time_s": round(t_detect, 2),
        "total_time_s": round(elapsed, 1),
        "panel_name": f"{scene_id}/mnf_rx_panels.png",
    }


def write_summary(results: list[dict]) -> None:
    """Write experiment summary markdown."""
    lines = [
        "# mnf-enmap: MNF Compression + Global RX on EnMAP L2A",
        "",
        "## Experiment",
        "",
        "- **Detector:** MNFCompressionDetector (MNF → GRX)",
        f"- **MNF components:** {N_COMPONENTS}",
        f"- **Band failure threshold:** {BAND_FAILURE_THRESHOLD * 100:.0f}%",
        f"- **Min band coverage:** {MIN_BAND_COVERAGE * 100:.0f}%",
        "- **Noise estimation:** spatial shift-difference (Green et al. 1988)",
        f"- **Scenes processed:** {len(results)}",
        "",
        "## Results",
        "",
        "| Scene | Cube | Bands (good/total) | MNF Comp. | Valid Pixels | Score Range | Median | p99 | Build (s) | Detect (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        bands_str = f"{r['n_good_bands']}/{r['n_total_bands']}"
        valid_str = f"{r['n_valid_pixels']:,}/{r['n_total_pixels']:,}"
        score_range = f"{r['score_min']}–{r['score_max']}"
        lines.append(
            f"| {r['scene_id'][:50]} | {r['cube_shape']} | {bands_str} | "
            f"{r['n_components']} | {valid_str} | {score_range} | "
            f"{r['score_median']} | {r['score_p99']} | {r['build_time_s']} | "
            f"{r['detect_time_s']} |"
        )

    lines += ["", "## Visualizations", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:60]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['panel_name']})")
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
