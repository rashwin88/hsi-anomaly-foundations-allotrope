"""
thermal-grx-landsat: Thermal Global RX anomaly detection on Landsat 9 B10.

Per scene:
  1. ThermalGRXDetector → GRX anomaly scores on single-band thermal
  2. Threshold at p99.5 → binary anomaly mask
  3. Plot anomalies as red transparent circles on black background
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
from app.utils.dataset_builder.landsat_dataset_builder import LandsatDataBuilder
from app.detectors.thermal_grx_detector import ThermalGRXDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("thermal-grx-landsat")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
THERMAL_PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads" / "thermal_3"

# ---- Experiment parameters ----
DETECTION_PERCENTILE = 99.5
CIRCLE_RADIUS = 8


def discover_thermal_files() -> list[Path]:
    """Find all Landsat B10 TIF files in thermal_3."""
    files = sorted(
        p for p in THERMAL_PAYLOADS.iterdir()
        if p.is_file() and p.name.endswith("_ST_B10.TIF")
    )
    logger.info("Found %d Landsat thermal B10 files.", len(files))
    return files


def find_qa_pixel(b10_path: Path) -> Path | None:
    """Look for matching QA_PIXEL.TIF alongside a B10 file."""
    # LC09_L2SP_132047_20240330_20240403_02_T1_ST_B10.TIF
    # → LC09_L2SP_132047_20240330_20240403_02_T1_QA_PIXEL.TIF
    qa_name = b10_path.name.replace("_ST_B10.TIF", "_QA_PIXEL.TIF")
    qa_path = b10_path.parent / qa_name
    return qa_path if qa_path.exists() else None


def write_anomaly_tif(
    out_path: Path,
    mask: np.ndarray,
    height: int,
    width: int,
    source_tif: Path = None,
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
    if source_tif is not None:
        with rasterio.open(str(source_tif)) as src:
            profile["crs"] = src.crs
            profile["transform"] = src.transform
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])


def save_detection_overlay(
    out_path: Path,
    thermal_img: np.ndarray,
    score_map: np.ndarray,
    spatial_mask: np.ndarray,
    det_mask: np.ndarray,
    scene_id: str,
    n_detections: int,
    n_valid_pixels: int,
) -> None:
    """
    Two-panel figure:
      Left:  Thermal GRX score map (inferno)
      Right: Red dots on black background
    """
    fig, (ax_scores, ax_overlay) = plt.subplots(1, 2, figsize=(18, 8))

    # -- Left panel: score map --
    display_map = score_map.copy()
    valid_scores = display_map[spatial_mask & ~np.isnan(display_map)]
    if len(valid_scores) > 0:
        vmin = np.percentile(valid_scores, 2)
        vmax = np.percentile(valid_scores, 98)
    else:
        vmin, vmax = 0.0, 1.0
    display_map[np.isnan(display_map)] = vmin
    display_map[~spatial_mask] = vmin

    im = ax_scores.imshow(display_map, cmap="inferno", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
    ax_scores.set_title("Thermal GRX Anomaly Scores", fontsize=11)
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
        f"{scene_id}\n"
        f"{n_valid_pixels:,} valid px | single-band thermal | "
        f"{n_detections:,} detections",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_single(b10_path: Path) -> dict:
    scene_id = b10_path.stem  # e.g. LC09_L2SP_142050_20240811_..._ST_B10
    scene_dir = OUTPUT_DIR / scene_id
    scene_dir.mkdir(exist_ok=True)
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    t0 = time.time()

    # 1. Build vendable
    qa_path = find_qa_pixel(b10_path)
    builder = LandsatDataBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(b10_path))
    )
    if qa_path is not None:
        vendable = builder.vend_dataset(provider_qa_pixel_source=str(qa_path))
        logger.info("Using QA_PIXEL mask from %s", qa_path.name)
    else:
        vendable = builder.vend_dataset()

    cube = vendable.normalized_thermal_cube
    validity = vendable.validity_cube
    C, H, W = cube.shape
    t_build = time.time() - t0
    logger.info("Cube shape: (%d, %d, %d)", C, H, W)

    # 2. Run Thermal GRX detection
    t1 = time.time()
    detector = ThermalGRXDetector(vendable)
    detector.fit()
    score_map = detector.detect(cube, validity)
    t_detect = time.time() - t1
    result = detector.result
    spatial_mask = result.spatial_mask

    # 3. Threshold at p99.5
    valid_scores = score_map[spatial_mask & ~np.isnan(score_map)]
    if len(valid_scores) == 0:
        logger.warning("No valid scores for %s — skipping.", scene_id)
        return None

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
    tif_path = scene_dir / "anomaly_thermal_grx.tif"
    write_anomaly_tif(tif_path, det_mask, H, W, source_tif=b10_path)
    logger.info("Wrote %s", tif_path.name)

    # 5. Save raw thermal visualization
    thermal_img = cube[0].copy()
    thermal_img[~spatial_mask] = np.nan

    fig_raw, ax_raw = plt.subplots(1, 1, figsize=(10, 8))
    valid_temps = thermal_img[spatial_mask]
    t_vmin = np.percentile(valid_temps, 2)
    t_vmax = np.percentile(valid_temps, 98)
    im_raw = ax_raw.imshow(thermal_img, cmap="inferno", vmin=t_vmin, vmax=t_vmax)
    ax_raw.set_title(f"{scene_id}\nSurface Temperature (°C)", fontsize=11)
    ax_raw.axis("off")
    fig_raw.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04, label="°C")
    fig_raw.tight_layout()
    raw_path = scene_dir / "raw_thermal.png"
    fig_raw.savefig(str(raw_path), dpi=150, bbox_inches="tight")
    plt.close(fig_raw)
    logger.info("Saved: %s", raw_path.name)

    # 6. Save detection overlay
    overlay_path = scene_dir / "detection_overlay.png"
    save_detection_overlay(
        overlay_path, thermal_img, score_map, spatial_mask, det_mask,
        scene_id, n_detections, result.n_valid_pixels,
    )
    logger.info("Saved: %s", overlay_path.name)

    elapsed = time.time() - t0

    return {
        "scene_id": scene_id,
        "cube_shape": f"{C}x{H}x{W}",
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
        "tif_name": f"{scene_id}/anomaly_thermal_grx.tif",
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# thermal-grx-landsat: Thermal GRX Anomaly Detection on Landsat 9",
        "",
        "## Experiment",
        "",
        "- **Detector:** ThermalGRXDetector (single-band GRX)",
        f"- **Detection threshold:** p{DETECTION_PERCENTILE}",
        f"- **Scenes processed:** {len(results)}",
        "",
        "## Results",
        "",
        "| Scene | Shape | Valid Px | Detections | Threshold | Median | p99 | p99.5 | Max | Time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in results:
        lines.append(
            f"| {r['scene_id'][:50]} | {r['cube_shape']} | "
            f"{r['n_valid_pixels']:,} | "
            f"{r['n_detections']:,} | {r['threshold']} | "
            f"{r['score_median']} | {r['score_p99']} | "
            f"{r['score_p995']} | {r['score_max']} | {r['total_time_s']} |"
        )

    lines += ["", "## Detection Overlays", ""]
    for r in results:
        lines.append(f"### {r['scene_id']}")
        lines.append(f"- Detections: {r['n_detections']:,}")
        lines.append(f"- GeoTIFF: [{r['tif_name']}]({r['tif_name']})")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['overlay_name']})")
        lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", summary_path)


def main():
    files = discover_thermal_files()
    if not files:
        logger.error("No Landsat thermal B10 files found.")
        return

    results = []
    for f in files:
        try:
            result = run_single(f)
            if result is not None:
                results.append(result)
        except Exception as e:
            logger.error("FAILED on %s: %s", f.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes processed.", len(results), len(files))


if __name__ == "__main__":
    main()
