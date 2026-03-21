"""
Generate band mosaic visualizations for all PRISMA and EnMAP scenes
in tests/test_payloads/.
"""

import gc
import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.detectors.global_rx_detector import GlobalRXDetector
from app.utils.visualization.hyperspectral_visualizer import plot_band_mosaic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("band-mosaics")

PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads"
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR.mkdir(exist_ok=True)


def process_enmap(folder: Path) -> None:
    scene_id = folder.name
    out_path = OUTPUT_DIR / f"{scene_id}_band_mosaic.png"
    logger.info("EnMAP: %s", scene_id)

    vendable = EnmapDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(folder))
    ).vend_dataset()

    detector = GlobalRXDetector(vendable)
    detector.fit()

    fig = plot_band_mosaic(
        vendable,
        good_band_indices=detector._good_indices,
        title=scene_id[:60],
        save_path=out_path,
    )
    plt.close(fig)
    del vendable, detector, fig
    gc.collect()


def process_prisma(he5_path: Path) -> None:
    scene_id = he5_path.stem
    out_path = OUTPUT_DIR / f"{scene_id}_band_mosaic.png"
    logger.info("PRISMA: %s", scene_id)

    vendable = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    ).vend_dataset()

    detector = GlobalRXDetector(vendable)
    detector.fit()

    fig = plot_band_mosaic(
        vendable,
        good_band_indices=detector._good_indices,
        title=scene_id[:60],
        save_path=out_path,
    )
    plt.close(fig)
    del vendable, detector, fig
    gc.collect()


def main():
    t0 = time.time()

    # EnMAP scenes
    enmap_folders = sorted(
        p for p in PAYLOADS.iterdir()
        if p.is_dir() and p.name.startswith("ENMAP")
    )
    logger.info("Found %d EnMAP scenes", len(enmap_folders))
    for folder in enmap_folders:
        try:
            process_enmap(folder)
        except Exception as e:
            logger.error("FAILED EnMAP %s: %s", folder.name, e, exc_info=True)

    # PRISMA scenes (find all .he5 files recursively, deduplicate by filename)
    he5_files = sorted(PAYLOADS.rglob("*.he5"))
    # Deduplicate by stem (same scene may appear in multiple sets)
    seen = set()
    unique_he5 = []
    for p in he5_files:
        if p.stem not in seen:
            seen.add(p.stem)
            unique_he5.append(p)
    logger.info("Found %d unique PRISMA scenes", len(unique_he5))

    for he5_path in unique_he5:
        try:
            process_prisma(he5_path)
        except Exception as e:
            logger.error("FAILED PRISMA %s: %s", he5_path.name, e, exc_info=True)

    elapsed = time.time() - t0
    total = len(enmap_folders) + len(unique_he5)
    logger.info("Done. %d scenes processed in %.1fs", total, elapsed)


if __name__ == "__main__":
    main()
