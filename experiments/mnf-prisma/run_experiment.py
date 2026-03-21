"""
mnf-prisma: Run MNFCompressionDetector on all PRISMA HE5 files in test_payloads.

Applies MNF dimensionality reduction (n → n_components bands) before
computing Global RX scores.  Deduplicates by filename, builds vendables
via PrismaDatasetBuilder, runs detection, saves visualizations, and
writes a summary markdown.
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
from app.detectors.mnf_compression_detector import MNFCompressionDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mnf-prisma")

OUTPUT_DIR = Path(__file__).resolve().parent

# ---- Experiment parameters ----
N_COMPONENTS = 15
BAND_FAILURE_THRESHOLD = 0.05
MIN_BAND_COVERAGE = 0.95


def discover_he5_files() -> list[Path]:
    """Find all .he5 files under test_payloads, deduplicated by filename."""
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
    """Build vendable, run MNF-RX, save visualization. Returns summary dict."""
    scene_id = he5_path.stem
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    # Build vendable
    t0 = time.time()
    builder = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    )
    vendable = builder.vend_dataset()
    t_build = time.time() - t0

    # Run MNF compression + GRX
    t1 = time.time()
    detector = MNFCompressionDetector(vendable)
    detector.fit(
        n_components=N_COMPONENTS,
        band_failure_threshold=BAND_FAILURE_THRESHOLD,
        min_band_coverage=MIN_BAND_COVERAGE,
    )
    score_map = detector.detect(
        vendable.normalized_hyperspectral_cube,
        vendable.validity_cube,
    )
    t_detect = time.time() - t1

    result = detector.result

    # Save 3-panel visualization (mask, scores, eigenvalue spectrum)
    fig = result.visualize()
    fig_path = OUTPUT_DIR / f"{scene_id}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", fig_path.name)

    valid_scores = score_map[~np.isnan(score_map)]

    return {
        "scene_id": scene_id,
        "path": str(he5_path),
        "cube_shape": vendable.normalized_hyperspectral_cube.shape,
        "n_total_bands": vendable.normalized_hyperspectral_cube.shape[0],
        "n_good_bands": result.n_good_bands,
        "n_components": result.n_components,
        "n_valid_pixels": result.n_valid_pixels,
        "n_total_pixels": score_map.shape[0] * score_map.shape[1],
        "pixel_band_ratio": round(result.n_valid_pixels / result.n_good_bands, 1),
        "score_min": round(float(valid_scores.min()), 4),
        "score_max": round(float(valid_scores.max()), 4),
        "score_median": round(float(np.median(valid_scores)), 4),
        "score_p99": round(float(np.percentile(valid_scores, 99)), 4),
        "build_time_s": round(t_build, 2),
        "detect_time_s": round(t_detect, 2),
        "fig_name": f"{scene_id}.png",
    }


def write_summary(results: list[dict]) -> None:
    """Write experiment summary markdown."""
    lines = [
        "# mnf-prisma: MNF Compression + Global RX on PRISMA Test Payloads",
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
        "| Scene | Cube Shape | Bands (good/total) | MNF Comp. | Valid Pixels | Score Range | Median | p99 | Build (s) | Detect (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        shape_str = f"`{r['cube_shape']}`"
        bands_str = f"{r['n_good_bands']}/{r['n_total_bands']}"
        valid_str = f"{r['n_valid_pixels']:,}/{r['n_total_pixels']:,}"
        score_range = f"{r['score_min']}–{r['score_max']}"
        lines.append(
            f"| {r['scene_id'][:40]} | {shape_str} | {bands_str} | {r['n_components']} | "
            f"{valid_str} | {score_range} | {r['score_median']} | {r['score_p99']} | "
            f"{r['build_time_s']} | {r['detect_time_s']} |"
        )

    lines += ["", "## Visualizations", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:40]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['fig_name']})")
        lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", summary_path)


def main():
    he5_files = discover_he5_files()
    if not he5_files:
        logger.error("No HE5 files found.")
        return

    results = []
    for path in he5_files:
        try:
            result = run_single(path)
            results.append(result)
        except Exception as e:
            logger.error("FAILED on %s: %s", path.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes completed.", len(results), len(he5_files))


if __name__ == "__main__":
    main()
