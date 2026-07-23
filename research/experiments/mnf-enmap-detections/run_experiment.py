"""
mnf-enmap-detections: MNF Compression + GRX anomaly detection on EnMAP.

Per scene:
  1. MNFCompressionDetector (5 components) → GRX anomaly scores
  2. Threshold at p99.5 → binary anomaly mask
  3. Plot anomalies as red transparent circles over a false-colour image
  4. Write binary GeoTIFF (1=anomaly, 0=background) with CRS from source
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.detectors.mnf_compression_detector import MNFCompressionDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("mnf-enmap-detections")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
ENMAP_PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads"

# ---- Experiment parameters ----
N_COMPONENTS = 3
BAND_FAILURE_THRESHOLD = 0.05
MIN_BAND_COVERAGE = 0.95
DETECTION_PERCENTILE = 99.5
CIRCLE_RADIUS = 8


def discover_enmap_folders() -> list[Path]:
    folders = sorted(
        p for p in ENMAP_PAYLOADS.iterdir()
        if p.is_dir() and p.name.startswith("ENMAP")
    )
    logger.info("Found %d EnMAP scene folders.", len(folders))
    return folders


def make_false_colour(cube: np.ndarray, validity: np.ndarray,
                      band_wavelengths: list[float]) -> np.ndarray:
    """
    Build a false-colour RGB from the cube using bands near 850/650/550 nm.

    Returns (H, W, 3) float in [0, 1].
    """
    wl = np.array(band_wavelengths)
    targets = [850.0, 650.0, 550.0]  # NIR, Red, Green
    indices = [int(np.argmin(np.abs(wl - t))) for t in targets]

    H, W = cube.shape[1], cube.shape[2]
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for ch, idx in enumerate(indices):
        band = cube[idx].astype(np.float32).copy()
        band_valid = validity[idx].astype(bool)
        # Percentile stretch on valid pixels
        vals = band[band_valid]
        if len(vals) > 0:
            lo = np.percentile(vals, 2)
            hi = np.percentile(vals, 98)
            if hi > lo:
                band = (band - lo) / (hi - lo)
            else:
                band = np.zeros_like(band)
        band = np.clip(band, 0, 1)
        band[~band_valid] = 0
        rgb[:, :, ch] = band

    return rgb


def write_anomaly_tif(
    out_path: Path,
    mask: np.ndarray,
    height: int,
    width: int,
    source_folder: Path = None,
) -> None:
    """Write a single-band uint8 GeoTIFF (1=anomaly, 0=background)."""
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": height,
        "width": width,
        "compress": "deflate",
    }
    if source_folder is not None:
        spec_tifs = list(source_folder.glob("*SPECTRAL_IMAGE.TIF"))
        if spec_tifs:
            with rasterio.open(spec_tifs[0]) as src:
                profile["crs"] = src.crs
                profile["transform"] = src.transform
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])


def save_detection_overlay(
    out_path: Path,
    rgb: np.ndarray,
    score_map: np.ndarray,
    spatial_mask: np.ndarray,
    det_mask: np.ndarray,
    scene_id: str,
    n_detections: int,
    n_components: int,
    n_good_bands: int,
    n_valid_pixels: int,
) -> None:
    """
    Two-panel figure:
      Left:  MNF-GRX score map (inferno)
      Right: False-colour image with red transparent circles on anomalies
    """
    fig, (ax_scores, ax_overlay) = plt.subplots(1, 2, figsize=(18, 8))

    # -- Left panel: score map --
    display_map = score_map.copy()
    valid_scores = display_map[spatial_mask & ~np.isnan(display_map)]
    vmin = np.percentile(valid_scores, 2)
    vmax = np.percentile(valid_scores, 98)
    display_map[np.isnan(display_map)] = vmin
    display_map[~spatial_mask] = vmin

    im = ax_scores.imshow(display_map, cmap="inferno", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
    ax_scores.set_title("MNF-GRX Anomaly Scores", fontsize=11)
    ax_scores.axis("off")
    fig.colorbar(im, ax=ax_scores, fraction=0.046, pad=0.04)

    # -- Right panel: red dots on black background --
    H_img, W_img = det_mask.shape
    ax_overlay.set_xlim(0, W_img)
    ax_overlay.set_ylim(H_img, 0)
    ax_overlay.set_facecolor("black")
    ax_overlay.set_aspect("equal")

    anom_rows, anom_cols = np.where(det_mask.astype(bool))
    for r, c in zip(anom_rows, anom_cols):
        circle = Circle(
            (c, r), radius=CIRCLE_RADIUS,
            edgecolor="red", facecolor="red", alpha=0.35,
            linewidth=0.8,
        )
        ax_overlay.add_patch(circle)

    ax_overlay.set_title(
        f"Detections (p{DETECTION_PERCENTILE} threshold): "
        f"{n_detections:,} anomalous pixels",
        fontsize=11,
    )
    ax_overlay.axis("off")

    fig.suptitle(
        f"{scene_id[:60]}\n"
        f"{n_valid_pixels:,} valid px | {n_good_bands} bands → "
        f"{n_components} MNF components | {n_detections:,} detections",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single(enmap_folder: Path) -> dict:
    scene_id = enmap_folder.name
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(exist_ok=True)
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    t0 = time.time()

    # 1. Build vendable
    vendable = EnmapDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(enmap_folder))
    ).vend_dataset()

    cube = vendable.normalized_hyperspectral_cube
    validity = vendable.validity_cube
    _, H, W = cube.shape
    t_build = time.time() - t0

    # 2. Run MNF-GRX detection
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
    spatial_mask = result.spatial_mask

    # 3. Threshold at p99.5
    valid_scores = score_map[spatial_mask & ~np.isnan(score_map)]
    threshold = np.percentile(valid_scores, DETECTION_PERCENTILE)
    det_mask = (
        (score_map >= threshold) & spatial_mask & ~np.isnan(score_map)
    ).astype(np.uint8)
    n_detections = int(det_mask.sum())
    logger.info(
        "Threshold=%.4f (p%.1f) → %d detections",
        threshold, DETECTION_PERCENTILE, n_detections,
    )

    # 4. Write GeoTIFF
    tif_path = scene_dir / "anomaly_mnf_rx.tif"
    write_anomaly_tif(tif_path, det_mask, H, W, source_folder=enmap_folder)
    logger.info("Wrote %s", tif_path.name)

    # 5. Build false-colour image and save overlay
    rgb = make_false_colour(cube, validity, vendable.band_cw_order)
    overlay_path = scene_dir / "detection_overlay.png"
    save_detection_overlay(
        overlay_path, rgb, score_map, spatial_mask, det_mask,
        scene_id, n_detections, result.n_components,
        result.n_good_bands, result.n_valid_pixels,
    )
    logger.info("Saved: %s", overlay_path.name)

    elapsed = time.time() - t0

    return {
        "scene_id": scene_id,
        "cube_shape": f"{cube.shape[0]}x{H}x{W}",
        "n_good_bands": result.n_good_bands,
        "n_components": result.n_components,
        "n_valid_pixels": result.n_valid_pixels,
        "n_detections": n_detections,
        "threshold": round(float(threshold), 4),
        "score_median": round(float(np.median(valid_scores)), 4),
        "score_p99": round(float(np.percentile(valid_scores, 99)), 4),
        "score_p995": round(float(np.percentile(valid_scores, 99.5)), 4),
        "score_max": round(float(valid_scores.max()), 4),
        "build_time_s": round(t_build, 2),
        "detect_time_s": round(t_detect, 2),
        "total_time_s": round(elapsed, 1),
        "overlay_name": f"{scene_id}/detection_overlay.png",
        "tif_name": f"{scene_id}/anomaly_mnf_rx.tif",
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# mnf-enmap-detections: MNF-GRX Anomaly Detection on EnMAP",
        "",
        "## Experiment",
        "",
        "- **Detector:** MNFCompressionDetector (MNF → GRX)",
        f"- **MNF components:** {N_COMPONENTS}",
        f"- **Band failure threshold:** {BAND_FAILURE_THRESHOLD * 100:.0f}%",
        f"- **Detection threshold:** p{DETECTION_PERCENTILE}",
        f"- **Scenes processed:** {len(results)}",
        "",
        "## Results",
        "",
        "| Scene | Bands | MNF | Valid Px | Detections | Threshold | Median | p99 | p99.5 | Max | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['scene_id'][:45]} | {r['n_good_bands']} | "
            f"{r['n_components']} | {r['n_valid_pixels']:,} | "
            f"{r['n_detections']:,} | {r['threshold']} | "
            f"{r['score_median']} | {r['score_p99']} | "
            f"{r['score_p995']} | {r['score_max']} | {r['total_time_s']} |"
        )

    lines += ["", "## Detection Overlays", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:60]}")
        lines.append(f"- Detections: {r['n_detections']:,}")
        lines.append(f"- GeoTIFF: [{r['tif_name']}]({r['tif_name']})")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['overlay_name']})")
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
