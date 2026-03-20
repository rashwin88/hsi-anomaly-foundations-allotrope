"""
rx-had-exp2: Run GlobalRXDetector on destriped PRISMA HE5 files.

Pipeline: HE5 → PrismaDatasetBuilder → CombinedDestriper → GlobalRXDetector
Saves RX visualizations and a summary comparing raw vs destriped scores.
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

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rx-had-exp2")

OUTPUT_DIR = Path(__file__).resolve().parent


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
    scene_id = he5_path.stem
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    # Build vendable
    builder = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    )
    vendable = builder.vend_dataset()
    cube = vendable.normalized_hyperspectral_cube
    validity = vendable.validity_cube
    band_wavelengths = vendable.band_cw_order

    # Destripe
    t0 = time.time()
    destriper = CombinedDestriper()
    destriped_cube = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=band_wavelengths,
    )
    t_destripe = time.time() - t0

    # Run RX on destriped cube
    t1 = time.time()
    detector = GlobalRXDetector(vendable)
    detector.fit(band_failure_threshold=0.05)
    score_map = detector.detect(destriped_cube, validity)
    t_detect = time.time() - t1

    result = detector.result

    # Save visualization
    fig = result.visualize()
    fig_path = OUTPUT_DIR / f"{scene_id}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    valid_scores = score_map[~np.isnan(score_map)]

    return {
        "scene_id": scene_id,
        "n_good_bands": result.n_good_bands,
        "n_valid_pixels": result.n_valid_pixels,
        "score_min": round(float(valid_scores.min()), 4),
        "score_max": round(float(valid_scores.max()), 4),
        "score_median": round(float(np.median(valid_scores)), 4),
        "destripe_time_s": round(t_destripe, 2),
        "detect_time_s": round(t_detect, 2),
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# rx-had-exp2: Global RX on Destriped PRISMA Scenes",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → CombinedDestriper (FFT + moment-matching with σ guard) → GlobalRXDetector",
        "",
        "## Results",
        "",
        "| Scene | Bands | Valid Pixels | Score Range | Median | Destripe (s) | RX (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:40]} | {r['n_good_bands']} | "
            f"{r['n_valid_pixels']:,} | {r['score_min']}–{r['score_max']} | "
            f"{r['score_median']} | {r['destripe_time_s']} | {r['detect_time_s']} |"
        )

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

    logger.info("Done. %d/%d scenes.", len(results), len(he5_files))


if __name__ == "__main__":
    main()
