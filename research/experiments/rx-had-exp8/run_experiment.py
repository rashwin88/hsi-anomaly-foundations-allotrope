"""
rx-had-exp8: GRX + LRX comparison — raw vs destriped with detection maps.

Per scene (9-panel figure):
  Row 1: Validity Mask | GRX Raw Scores | GRX Destriped Scores | GRX Det Raw | GRX Det Destriped
  Row 2:    (blank)    | LRX Raw Scores | LRX Destriped Scores | LRX Det Raw | LRX Det Destriped

Detection threshold: p99.5 of the respective score map.
Score maps use shared vmin/vmax per detector (raw+destriped combined) so comparisons are fair.
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
from app.detectors.local_rx_detector import LocalRXDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rx-had-exp8")
logging.getLogger("rx-had-exp8").setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
FAIL_THRESH = 0.05
DETECTION_PERCENTILE = 99.5
LRX_STRIDE = 2
LRX_OUTER = 25
LRX_INNER = 5


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


def _make_detection_mask(score_map, spatial_mask, percentile):
    """Binary mask: pixels above the given percentile among valid scores."""
    valid_scores = score_map[spatial_mask & ~np.isnan(score_map)]
    if len(valid_scores) == 0:
        return np.zeros_like(spatial_mask)
    thresh = np.percentile(valid_scores, percentile)
    return (score_map >= thresh) & spatial_mask & ~np.isnan(score_map)


def _detection_rgb(spatial_mask, det_mask):
    """RGB image: dark gray=invalid, white=valid, red=detection."""
    img = np.full(spatial_mask.shape + (3,), fill_value=0.15)
    img[spatial_mask] = [1.0, 1.0, 1.0]
    img[det_mask] = [1.0, 0.0, 0.0]
    return img


def save_panels(
    scene_id: str,
    spatial_mask: np.ndarray,
    grx_raw: np.ndarray,
    grx_des: np.ndarray,
    lrx_raw: np.ndarray,
    lrx_des: np.ndarray,
    n_valid: int,
    n_good_bands: int,
) -> Path:
    """2×5 panel figure: validity + score maps + detection maps."""
    fig, axes = plt.subplots(2, 5, figsize=(28, 10))
    fig.suptitle(
        f"{scene_id[:60]}\n"
        f"{n_valid:,} valid px · {n_good_bands} bands · "
        f"detection threshold: p{DETECTION_PERCENTILE}",
        fontsize=10,
    )

    cmap = plt.cm.inferno.copy()
    cmap.set_bad("lightgray")

    # --- Shared scales per detector (raw + destriped combined) ---
    grx_raw_v = grx_raw[spatial_mask]
    grx_des_v = grx_des[spatial_mask]
    grx_combined = np.concatenate([grx_raw_v, grx_des_v])
    grx_lo, grx_hi = np.percentile(grx_combined, [2, 99.5])

    lrx_raw_v = lrx_raw[spatial_mask & ~np.isnan(lrx_raw)]
    lrx_des_v = lrx_des[spatial_mask & ~np.isnan(lrx_des)]
    lrx_combined = np.concatenate([lrx_raw_v, lrx_des_v])
    lrx_lo, lrx_hi = np.percentile(lrx_combined, [2, 99.5])

    # Detection masks
    grx_det_raw = _make_detection_mask(grx_raw, spatial_mask, DETECTION_PERCENTILE)
    grx_det_des = _make_detection_mask(grx_des, spatial_mask, DETECTION_PERCENTILE)
    lrx_det_raw = _make_detection_mask(lrx_raw, spatial_mask, DETECTION_PERCENTILE)
    lrx_det_des = _make_detection_mask(lrx_des, spatial_mask, DETECTION_PERCENTILE)

    # ---- Row 0: GRX ----
    # (0,0) Validity mask
    axes[0, 0].imshow(spatial_mask, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("Validity Mask")
    axes[0, 0].axis("off")

    # (0,1) GRX raw scores
    im = axes[0, 1].imshow(
        np.where(spatial_mask, grx_raw, np.nan),
        cmap=cmap, vmin=grx_lo, vmax=grx_hi, interpolation="nearest",
    )
    axes[0, 1].set_title("GRX Scores — Raw")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04, extend="max")

    # (0,2) GRX destriped scores
    im = axes[0, 2].imshow(
        np.where(spatial_mask, grx_des, np.nan),
        cmap=cmap, vmin=grx_lo, vmax=grx_hi, interpolation="nearest",
    )
    axes[0, 2].set_title("GRX Scores — Destriped")
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04, extend="max")

    # (0,3) GRX detections raw
    axes[0, 3].imshow(_detection_rgb(spatial_mask, grx_det_raw), interpolation="nearest")
    axes[0, 3].set_title(f"GRX Det Raw ({int(grx_det_raw.sum()):,})")
    axes[0, 3].axis("off")

    # (0,4) GRX detections destriped
    axes[0, 4].imshow(_detection_rgb(spatial_mask, grx_det_des), interpolation="nearest")
    axes[0, 4].set_title(f"GRX Det Destriped ({int(grx_det_des.sum()):,})")
    axes[0, 4].axis("off")

    # ---- Row 1: LRX ----
    # (1,0) blank
    axes[1, 0].axis("off")

    # (1,1) LRX raw scores
    im = axes[1, 1].imshow(
        np.where(spatial_mask, lrx_raw, np.nan),
        cmap=cmap, vmin=lrx_lo, vmax=lrx_hi, interpolation="nearest",
    )
    axes[1, 1].set_title("LRX Scores — Raw")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, extend="max")

    # (1,2) LRX destriped scores
    im = axes[1, 2].imshow(
        np.where(spatial_mask, lrx_des, np.nan),
        cmap=cmap, vmin=lrx_lo, vmax=lrx_hi, interpolation="nearest",
    )
    axes[1, 2].set_title("LRX Scores — Destriped")
    axes[1, 2].axis("off")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04, extend="max")

    # (1,3) LRX detections raw
    axes[1, 3].imshow(_detection_rgb(spatial_mask, lrx_det_raw), interpolation="nearest")
    axes[1, 3].set_title(f"LRX Det Raw ({int(lrx_det_raw.sum()):,})")
    axes[1, 3].axis("off")

    # (1,4) LRX detections destriped
    axes[1, 4].imshow(_detection_rgb(spatial_mask, lrx_det_des), interpolation="nearest")
    axes[1, 4].set_title(f"LRX Det Destriped ({int(lrx_det_des.sum()):,})")
    axes[1, 4].axis("off")

    fig.tight_layout()
    out = OUTPUT_DIR / f"{scene_id}_panels.png"
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
    t_des = time.time()
    destriper = CombinedDestriper()
    destriped_cube = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=False,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )
    t_des = time.time() - t_des
    logger.info("Destriping done in %.1fs", t_des)

    # --- GRX ---
    grx = GlobalRXDetector(vendable)
    grx.fit(band_failure_threshold=FAIL_THRESH)

    t_grx = time.time()
    grx_raw_map = grx.detect(cube, validity)
    grx_des_map = grx.detect(destriped_cube, validity)
    t_grx = time.time() - t_grx

    spatial_mask = grx.result.spatial_mask
    n_good_bands = grx.result.n_good_bands
    n_valid = int(spatial_mask.sum())
    logger.info("GRX done in %.1fs | bands=%d | valid_px=%d", t_grx, n_good_bands, n_valid)

    # --- LRX ---
    lrx = LocalRXDetector(vendable)
    lrx.fit(band_failure_threshold=FAIL_THRESH)

    t_lrx = time.time()
    lrx_raw_map = lrx.detect(
        cube, validity,
        outer_window=LRX_OUTER, inner_window=LRX_INNER, stride=LRX_STRIDE,
    )
    lrx_des_map = lrx.detect(
        destriped_cube, validity,
        outer_window=LRX_OUTER, inner_window=LRX_INNER, stride=LRX_STRIDE,
    )
    t_lrx = time.time() - t_lrx
    logger.info("LRX done in %.1fs", t_lrx)

    # Save panels
    panel_path = save_panels(
        scene_id, spatial_mask,
        grx_raw_map, grx_des_map,
        lrx_raw_map, lrx_des_map,
        n_valid, n_good_bands,
    )

    elapsed = time.time() - t0
    logger.info("Scene done in %.1fs | fig=%s", elapsed, panel_path.name)

    def stats(arr, mask):
        v = arr[mask & ~np.isnan(arr)]
        if len(v) == 0:
            return {f"p{p}": np.nan for p in [50, 99, 99.5]}
        return {
            "median": round(float(np.median(v)), 2),
            "p99":    round(float(np.percentile(v, 99)), 2),
            "p99.5":  round(float(np.percentile(v, 99.5)), 2),
            "max":    round(float(v.max()), 2),
            "n_det":  int((v >= np.percentile(v, DETECTION_PERCENTILE)).sum()),
        }

    return {
        "scene_id":     scene_id,
        "n_good_bands": n_good_bands,
        "n_valid":      n_valid,
        "grx_raw":      stats(grx_raw_map, spatial_mask),
        "grx_des":      stats(grx_des_map, spatial_mask),
        "lrx_raw":      stats(lrx_raw_map, spatial_mask),
        "lrx_des":      stats(lrx_des_map, spatial_mask),
        "time_s":       round(elapsed, 1),
        "panel_name":   panel_path.name,
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# rx-had-exp8: GRX + LRX — Raw vs Destriped with Detection Maps",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → {raw, CombinedDestriper} → "
        "{GlobalRXDetector, LocalRXDetector(outer=25, inner=5, stride=2)} "
        f"→ score maps + detection at p{DETECTION_PERCENTILE}",
        "",
        "Score maps use shared vmin/vmax per detector (raw+destriped combined p2–p99.5, inferno cmap).",
        "",
        "## GRX Comparison",
        "",
        "| Scene | Bands | Valid Px | Raw Median | Raw p99.5 | Raw Max | Des Median | Des p99.5 | Des Max |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        g_r, g_d = r["grx_raw"], r["grx_des"]
        lines.append(
            f"| {r['scene_id'][:40]} | {r['n_good_bands']} | {r['n_valid']:,} | "
            f"{g_r['median']} | {g_r['p99.5']} | {g_r['max']} | "
            f"{g_d['median']} | {g_d['p99.5']} | {g_d['max']} |"
        )

    lines += [
        "",
        "## LRX Comparison",
        "",
        "| Scene | Raw Median | Raw p99.5 | Raw Max | Des Median | Des p99.5 | Des Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        l_r, l_d = r["lrx_raw"], r["lrx_des"]
        lines.append(
            f"| {r['scene_id'][:40]} | "
            f"{l_r['median']} | {l_r['p99.5']} | {l_r['max']} | "
            f"{l_d['median']} | {l_d['p99.5']} | {l_d['max']} |"
        )

    lines += [
        "",
        "## Detection Counts (p99.5 threshold)",
        "",
        "| Scene | GRX Raw | GRX Des | LRX Raw | LRX Des |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['scene_id'][:40]} | "
            f"{r['grx_raw']['n_det']:,} | {r['grx_des']['n_det']:,} | "
            f"{r['lrx_raw']['n_det']:,} | {r['lrx_des']['n_det']:,} |"
        )

    lines += ["", "## Panels", ""]
    for r in results:
        lines.append(f"### {r['scene_id'][:50]}")
        lines.append("")
        lines.append(f"![{r['scene_id']}]({r['panel_name']})")
        lines.append("")

    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))
    logger.info("Summary written to summary.md")


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
