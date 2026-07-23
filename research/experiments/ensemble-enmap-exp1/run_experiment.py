"""
ensemble-enmap-exp1: StatisticalEnsembler on all EnMAP L2A scenes.

Pipeline per scene:
  EnMAP folder → EnmapDatasetBuilder → CombinedDestriper (inside ensembler)
  → GRX + LRX → CDF-normalize → fuse (product / maximum / mean)
  → anomaly threshold at p99.5 of each fused map

Outputs per scene (in experiments/ensemble-enmap-exp1/<scene_id>/):
  - ensemble_panels.png    4-panel: GRX raw, LRX raw, GRX CDF-norm, fused (product)
  - anomaly_product.tif    Binary TIFF (uint8, 1=anomaly, 0=background) — product fusion
  - anomaly_maximum.tif    Binary TIFF — maximum fusion
  - anomaly_mean.tif       Binary TIFF — mean fusion

Global output:
  - summary.md             Per-scene stats and detection counts
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.enmap_dataset_builder import EnmapDatasetBuilder
from app.detectors.statistical_ensembler import StatisticalEnsembler
from app.models.anomaly_detection.ensemble_rx_result import FUSION_STRATEGIES

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("ensemble-enmap-exp1")
logger.setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
FAIL_THRESH = 0.05  # default — coverage-fraction mask handles EnMAP column defects
DETECTION_PERCENTILE = 99.5
LRX_STRIDE = 2
LRX_OUTER = 25
LRX_INNER = 5
DISPLAY_GAMMA = 3   # power-law stretch for CDF-norm panels (compresses background)

ENMAP_PAYLOADS = PROJECT_ROOT / "tests" / "test_payloads"


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def discover_enmap_folders() -> list[Path]:
    """Find all EnMAP L2A scene folders under the test payloads dir."""
    folders = sorted(
        p for p in ENMAP_PAYLOADS.iterdir()
        if p.is_dir() and p.name.startswith("ENMAP")
    )
    logger.info("Found %d EnMAP scene folders.", len(folders))
    return folders


def make_detection_mask(
    score_map: np.ndarray,
    spatial_mask: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Binary mask: 1 where score >= percentile among valid pixels, else 0."""
    valid_scores = score_map[spatial_mask & ~np.isnan(score_map)]
    if len(valid_scores) == 0:
        return np.zeros_like(spatial_mask, dtype=np.uint8)
    thresh = np.percentile(valid_scores, percentile)
    return (
        (score_map >= thresh) & spatial_mask & ~np.isnan(score_map)
    ).astype(np.uint8)


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
    # Copy CRS + transform from the source spectral image if available
    if source_folder is not None:
        spec_tifs = list(source_folder.glob("*SPECTRAL_IMAGE.TIF"))
        if spec_tifs:
            with rasterio.open(spec_tifs[0]) as src:
                profile["crs"] = src.crs
                profile["transform"] = src.transform
    with rasterio.open(str(out_path), "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])


def save_panels(
    scene_dir: Path,
    result,
    spatial_mask: np.ndarray,
    det_masks: dict[str, np.ndarray],
) -> Path:
    """6-panel figure: GRX raw, LRX raw, norm GRX, norm LRX, fused product, detections."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    cmap_raw = plt.cm.inferno.copy()
    cmap_raw.set_bad("lightgray")
    cmap_norm = plt.cm.inferno.copy()
    cmap_norm.set_bad("lightgray")

    def _show_raw(ax, data, title):
        """Raw score maps — auto-scaled percentile range."""
        valid = data[~np.isnan(data)]
        if valid.size == 0:
            ax.set_title(title)
            ax.axis("off")
            return
        vmin = np.percentile(valid, 2)
        vmax = np.percentile(valid, 98)
        im = ax.imshow(data, cmap=cmap_raw, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _show_norm(ax, data, title, gamma=DISPLAY_GAMMA):
        """CDF-norm / fused maps — gamma power-law stretch to compress background."""
        display = data.copy()
        valid_mask = ~np.isnan(display)
        display[valid_mask] = np.power(display[valid_mask], gamma)
        im = ax.imshow(
            display, cmap=cmap_norm, vmin=0, vmax=1, interpolation="nearest",
        )
        ax.set_title(f"{title} (γ={gamma})", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0: raw score maps + normalized GRX
    _show_raw(axes[0, 0], result.grx_result.rx_score_map, "GRX (raw)")
    _show_raw(axes[0, 1], result.lrx_result.lrx_score_map, "LRX (raw)")
    _show_norm(axes[0, 2], result.normalized_grx_map, "GRX (CDF-norm)")

    # Row 1: normalized LRX + fused product + detection overlay
    _show_norm(axes[1, 0], result.normalized_lrx_map, "LRX (CDF-norm)")
    _show_norm(axes[1, 1], result.fused_score_maps["product"], "Fused (product)")

    # Detection overlay: white=valid, red=detected, gray=invalid
    det_product = det_masks["product"]
    det_rgb = np.full(spatial_mask.shape + (3,), fill_value=0.15)
    det_rgb[spatial_mask] = [1.0, 1.0, 1.0]
    det_rgb[det_product.astype(bool)] = [1.0, 0.0, 0.0]
    axes[1, 2].imshow(det_rgb, interpolation="nearest")
    axes[1, 2].set_title(
        f"Detections — product ({int(det_product.sum()):,} px)", fontsize=10
    )
    axes[1, 2].axis("off")

    destrip_label = "destriped" if result.destriped else "raw"
    fig.suptitle(
        f"{scene_dir.name[:60]}\n"
        f"{result.n_valid_pixels:,} valid px | {result.n_good_bands} bands | "
        f"{destrip_label} | threshold: p{DETECTION_PERCENTILE}",
        fontsize=11,
    )
    fig.tight_layout()
    out = scene_dir / "ensemble_panels.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------------
# per-scene runner
# ------------------------------------------------------------------

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
    logger.info("Cube shape: %s", cube.shape)

    # 2. Create and fit ensembler
    ensembler = StatisticalEnsembler(vendable)
    ensembler.fit(band_failure_threshold=FAIL_THRESH)

    # 3. Run detection with destriping
    t_det = time.time()
    _ = ensembler.detect(
        cube,
        validity_mask=validity,
        destripe=True,
        outer_window=LRX_OUTER,
        inner_window=LRX_INNER,
        stride=LRX_STRIDE,
    )
    t_det = time.time() - t_det
    result = ensembler.result

    spatial_mask = result.spatial_mask
    n_valid = result.n_valid_pixels

    logger.info(
        "Detection done in %.1fs | bands=%d | valid_px=%d",
        t_det, result.n_good_bands, n_valid,
    )

    # 4. Build detection masks and write TIFFs for each strategy
    det_masks = {}
    for strategy in FUSION_STRATEGIES:
        fused_map = result.fused_score_maps[strategy]
        det = make_detection_mask(fused_map, spatial_mask, DETECTION_PERCENTILE)
        det_masks[strategy] = det

        tif_path = scene_dir / f"anomaly_{strategy}.tif"
        write_anomaly_tif(tif_path, det, H, W, source_folder=enmap_folder)
        logger.info("Wrote %s (%d detections)", tif_path.name, int(det.sum()))

    # 5. Save panel figure
    panel_path = save_panels(scene_dir, result, spatial_mask, det_masks)

    elapsed = time.time() - t0
    logger.info("Scene done in %.1fs | output=%s", elapsed, scene_dir)

    # 6. Collect stats
    def score_stats(score_map, mask):
        v = score_map[mask & ~np.isnan(score_map)]
        if len(v) == 0:
            return {"median": np.nan, "p99": np.nan, "p99.5": np.nan, "max": np.nan}
        return {
            "median": round(float(np.median(v)), 4),
            "p99": round(float(np.percentile(v, 99)), 4),
            "p99.5": round(float(np.percentile(v, 99.5)), 4),
            "max": round(float(np.max(v)), 4),
        }

    return {
        "scene_id": scene_id,
        "n_good_bands": result.n_good_bands,
        "n_valid": n_valid,
        "cube_shape": f"{cube.shape[0]}x{H}x{W}",
        "grx_stats": score_stats(result.grx_result.rx_score_map, spatial_mask),
        "lrx_stats": score_stats(result.lrx_result.lrx_score_map, spatial_mask),
        "fused_product_stats": score_stats(result.fused_score_maps["product"], spatial_mask),
        "fused_maximum_stats": score_stats(result.fused_score_maps["maximum"], spatial_mask),
        "fused_mean_stats": score_stats(result.fused_score_maps["mean"], spatial_mask),
        "det_product": int(det_masks["product"].sum()),
        "det_maximum": int(det_masks["maximum"].sum()),
        "det_mean": int(det_masks["mean"].sum()),
        "time_s": round(elapsed, 1),
        "panel_name": str(panel_path.relative_to(OUTPUT_DIR)),
    }


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------

def write_summary(results: list[dict]) -> None:
    lines = [
        "# ensemble-enmap-exp1: StatisticalEnsembler on EnMAP L2A",
        "",
        "## Pipeline",
        "",
        "EnMAP folder -> EnmapDatasetBuilder -> StatisticalEnsembler "
        "(CombinedDestriper -> GRX + LRX -> CDF-normalize -> fuse) "
        f"-> anomaly detection at p{DETECTION_PERCENTILE}",
        "",
        f"- Band failure threshold: {FAIL_THRESH}",
        f"- Spatial mask: coverage-fraction >= 0.95 (with band-mean fill for missing bands)",
        f"- Display gamma: {DISPLAY_GAMMA} (power-law stretch on CDF-norm panels)",
        f"- LRX: outer={LRX_OUTER}, inner={LRX_INNER}, stride={LRX_STRIDE}",
        f"- Detection percentile: {DETECTION_PERCENTILE}",
        f"- Fusion strategies: {', '.join(FUSION_STRATEGIES)}",
        "",
        "## Scene Overview",
        "",
        "| Scene | Cube | Bands | Valid Px | Time (s) |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:50]} | {r['cube_shape']} | "
            f"{r['n_good_bands']} | {r['n_valid']:,} | {r['time_s']} |"
        )

    # GRX stats
    lines += [
        "",
        "## GRX Score Statistics (destriped)",
        "",
        "| Scene | Median | p99 | p99.5 | Max |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        s = r["grx_stats"]
        lines.append(
            f"| {r['scene_id'][:50]} | {s['median']} | {s['p99']} | {s['p99.5']} | {s['max']} |"
        )

    # LRX stats
    lines += [
        "",
        "## LRX Score Statistics (destriped)",
        "",
        "| Scene | Median | p99 | p99.5 | Max |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        s = r["lrx_stats"]
        lines.append(
            f"| {r['scene_id'][:50]} | {s['median']} | {s['p99']} | {s['p99.5']} | {s['max']} |"
        )

    # Fused score stats
    for strategy in FUSION_STRATEGIES:
        key = f"fused_{strategy}_stats"
        lines += [
            "",
            f"## Fused Score Statistics — {strategy}",
            "",
            "| Scene | Median | p99 | p99.5 | Max |",
            "|---|---|---|---|---|",
        ]
        for r in results:
            s = r[key]
            lines.append(
                f"| {r['scene_id'][:50]} | {s['median']} | {s['p99']} | {s['p99.5']} | {s['max']} |"
            )

    # Detection counts
    lines += [
        "",
        f"## Detection Counts (p{DETECTION_PERCENTILE} threshold)",
        "",
        "| Scene | Product | Maximum | Mean |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:50]} | {r['det_product']:,} | "
            f"{r['det_maximum']:,} | {r['det_mean']:,} |"
        )

    # Panel links
    lines += ["", "## Panels", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:60]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['panel_name']})")
        lines.append("")

    summary_path = OUTPUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    logger.info("Summary written to %s", summary_path)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

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
