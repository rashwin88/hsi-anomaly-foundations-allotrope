"""
rx-had-exp5: Destripe → GlobalRX with 20 partitioned probe bands (10 VNIR + 10 SWIR).

Pipeline: HE5 → PrismaDatasetBuilder → CombinedDestriper (family-partitioned probes)
          → GlobalRXDetector (rx-had-v1)

Per-scene output:
  - RX visualization PNG
  - Detailed stats in summary.md:
      band filtering breakdown, FFT angles + strengths, MM guard result,
      spatial mask stats, score distribution (p2/p25/median/p75/p98/p99/p99.9/max),
      outlier pixel count, per-band stripe reduction (3 representative bands)
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
from app.utils.data_transformations.spectral_band_filter import (
    SpectralBandFilter, DEFAULT_EXCLUSION_RANGES,
)
from app.utils.data_transformations.composite_destriper import CombinedDestriper
from app.detectors.global_rx_detector import GlobalRXDetector

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rx-had-exp5")
logging.getLogger("rx-had-exp5").setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).resolve().parent
FAIL_THRESH = 0.05


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


def col_mean_sigma(band: np.ndarray, validity_2d: np.ndarray) -> float:
    col_means = []
    for j in range(band.shape[1]):
        mask = validity_2d[:, j].astype(bool)
        if mask.sum() > 0:
            col_means.append(float(band[mask, j].mean()))
    return float(np.std(col_means)) if col_means else 0.0


def band_filter_stats(B, band_flags, wavelengths, validity, any_valid, n_spatial) -> dict:
    """Replicate the two-stage band filtering and return counts."""
    # Stage 1a: flags
    after_flags = [i for i, f in enumerate(band_flags) if f == 1]
    n_dropped_flags = B - len(after_flags)

    # Stage 1b: wavelength
    sbf = SpectralBandFilter(wavelengths, band_flags, DEFAULT_EXCLUSION_RANGES)
    good_1b = sbf.get_good_band_indices()
    n_dropped_wl = len(after_flags) - len(good_1b)

    # Stage 2: per-pixel failure
    good_final = []
    dropped_s2 = []
    for i in good_1b:
        n_invalid = n_spatial - int(validity[i][any_valid].sum())
        rate = n_invalid / n_spatial
        if rate > FAIL_THRESH:
            dropped_s2.append((i, rate))
        else:
            good_final.append(i)

    return {
        "input": B,
        "dropped_flags": n_dropped_flags,
        "dropped_wl": n_dropped_wl,
        "dropped_px_fail": len(dropped_s2),
        "final": len(good_final),
        "dropped_px_detail": dropped_s2,
    }


def stripe_check(cube, destriped, validity, wavelengths, good_bands) -> list[dict]:
    """Column-mean σ before/after for 3 representative bands."""
    def pick(target):
        return min(good_bands, key=lambda i: abs(wavelengths[i] - target))

    results = []
    for target, label in [(600, "VNIR ~600nm"), (1000, "boundary ~1000nm"), (2000, "SWIR ~2000nm")]:
        b = pick(target)
        v2d = validity[b].astype(bool)
        s_before = col_mean_sigma(cube[b], v2d)
        s_after  = col_mean_sigma(destriped[b], v2d)
        pct = (s_after - s_before) / s_before * 100 if s_before > 0 else 0.0
        results.append({
            "band": b,
            "wavelength": wavelengths[b],
            "label": label,
            "sigma_before": s_before,
            "sigma_after": s_after,
            "delta_pct": pct,
        })
    return results


def run_single(he5_path: Path) -> dict:
    scene_id = he5_path.stem
    logger.info("=" * 60)
    logger.info("Processing: %s", scene_id)

    t_start = time.time()

    builder = PrismaDatasetBuilder(
        file_source_configuration=FileSourceConfig(source_path=str(he5_path))
    )
    vendable = builder.vend_dataset()

    cube                  = vendable.normalized_hyperspectral_cube
    validity              = vendable.validity_cube
    wavelengths           = vendable.band_cw_order
    band_flags            = vendable.band_validity_by_position
    spectral_family_order = vendable.spectral_family_order

    B, H, W = cube.shape
    any_valid = validity.any(axis=0)
    n_spatial = int(any_valid.sum())

    # Band filtering stats (audit only, not re-running the detector logic)
    bf = band_filter_stats(B, band_flags, wavelengths, validity, any_valid, n_spatial)

    # Destripe (with diagnostics=True so we capture FFT angles + MM σ)
    t_ds = time.time()
    destriper = CombinedDestriper()
    destriped = destriper.transform(
        cube,
        validity_mask=validity,
        diagnostics=True,
        band_wavelengths=wavelengths,
        spectral_family_order=spectral_family_order,
    )
    t_ds = time.time() - t_ds

    diag = destriper.diagnostics
    fft_diag = diag.fft_diagnostics

    # FFT angles info
    detected_angles = diag.detected_angles or []
    angle_strengths = (fft_diag.angle_strengths or []) if fft_diag else []
    angle_rps       = (fft_diag.angle_radial_preserves or []) if fft_diag else []

    # MM σ progression
    sigma_orig     = diag.sigma_original
    sigma_fft      = diag.sigma_after_fft
    sigma_combined = diag.sigma_after_combined

    # RX on destriped cube
    t_rx = time.time()
    detector = GlobalRXDetector(vendable)
    detector.fit(band_failure_threshold=FAIL_THRESH)
    score_map = detector.detect(destriped, validity)
    t_rx = time.time() - t_rx

    result = detector.result
    spatial_mask = result.spatial_mask
    good_bands   = result.good_band_indices

    valid_scores = score_map[~np.isnan(score_map)]
    n_valid = int(spatial_mask.sum())

    # Score percentiles
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

    # Outlier counts at various thresholds
    p99 = percs["p99"]
    p999 = percs["p999"]
    n_outliers_p99  = int(np.sum(valid_scores > p99))
    n_outliers_p999 = int(np.sum(valid_scores > p999))

    # Per-band stripe check
    stripe = stripe_check(cube, destriped, validity, wavelengths, good_bands)

    # Save RX visualization
    fig = result.visualize()
    fig_path = OUTPUT_DIR / f"{scene_id}.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

    t_total = time.time() - t_start
    logger.info(
        "Done in %.1fs | bands=%d | valid_px=%d | angles=%s | p99=%.1f",
        t_total, bf["final"], n_valid, detected_angles, p99,
    )

    return {
        "scene_id": scene_id,
        "cube_shape": cube.shape,
        # Band filtering
        "bf": bf,
        # FFT
        "detected_angles": detected_angles,
        "angle_strengths": angle_strengths,
        "angle_rps": angle_rps,
        # MM σ
        "sigma_orig": sigma_orig,
        "sigma_fft": sigma_fft,
        "sigma_combined": sigma_combined,
        # Spatial
        "n_valid": n_valid,
        "n_total": H * W,
        "px_band_ratio": round(n_valid / bf["final"], 1) if bf["final"] else 0,
        # Scores
        "percs": percs,
        "n_outliers_p99": n_outliers_p99,
        "n_outliers_p999": n_outliers_p999,
        # Stripe
        "stripe": stripe,
        # Timing
        "t_destripe_s": round(t_ds, 1),
        "t_rx_s": round(t_rx, 1),
        "t_total_s": round(t_total, 1),
        "fig_name": fig_path.name,
    }


def write_summary(results: list[dict]) -> None:
    lines = [
        "# rx-had-exp5: Destripe → Global RX — 20 Partitioned Probes (10 VNIR + 10 SWIR)",
        "",
        "## Pipeline",
        "",
        "HE5 → PrismaDatasetBuilder → **CombinedDestriper** (20 probes: 10 VNIR + 10 SWIR, FFT notch + MM with σ guard) → **GlobalRXDetector** (rx-had-v1)",
        "",
        "---",
        "",
    ]

    for r in results:
        bf = r["bf"]
        lines += [
            f"## {r['scene_id'][:50]}",
            "",
            f"**Cube:** `{r['cube_shape']}`  |  "
            f"**Destripe:** {r['t_destripe_s']}s  |  "
            f"**RX:** {r['t_rx_s']}s  |  "
            f"**Total:** {r['t_total_s']}s",
            "",
            "### Band Filtering",
            "",
            f"| Stage | Dropped | Surviving |",
            f"|---|---|---|",
            f"| Input | — | {bf['input']} |",
            f"| 1a — validity flags | {bf['dropped_flags']} | {bf['input'] - bf['dropped_flags']} |",
            f"| 1b — wavelength exclusion | {bf['dropped_wl']} | {bf['input'] - bf['dropped_flags'] - bf['dropped_wl']} |",
            f"| 2 — pixel failure rate (>{FAIL_THRESH*100:.0f}%) | {bf['dropped_px_fail']} | **{bf['final']}** |",
            "",
        ]
        if bf["dropped_px_detail"]:
            lines.append("Bands dropped in stage 2 (index, wavelength, failure rate):")
            lines.append("")
            for idx, rate in bf["dropped_px_detail"]:
                lines.append(f"- Band {idx} — {rate*100:.1f}% failure")
            lines.append("")

        lines += [
            "### FFT Destriper",
            "",
        ]
        if r["detected_angles"]:
            lines.append("| Angle | Strength (σ) | Radial Preserve |")
            lines.append("|---|---|---|")
            for ang, stre, rp in zip(r["detected_angles"], r["angle_strengths"], r["angle_rps"]):
                lines.append(f"| {ang:.1f}° | {stre:.1f}σ | {rp} |")
        else:
            lines.append("No stripe angles detected.")
        lines.append("")

        # MM σ progression
        def fmt(v):
            return f"{v:.6f}" if v is not None else "N/A"

        lines += [
            "### Moment Matching σ Guard",
            "",
            f"| Stage | Column-mean σ (rep band) |",
            f"|---|---|",
            f"| Original | {fmt(r['sigma_orig'])} |",
            f"| After FFT | {fmt(r['sigma_fft'])} |",
            f"| After MM | {fmt(r['sigma_combined'])} |",
            "",
        ]

        if r["sigma_fft"] is not None and r["sigma_combined"] is not None:
            if abs(r["sigma_combined"] - r["sigma_fft"]) < 1e-9:
                lines.append("σ guard fired — moment-matching skipped (would have increased σ).")
            else:
                delta = (r["sigma_combined"] - r["sigma_fft"]) / r["sigma_fft"] * 100
                lines.append(f"Moment-matching applied — σ change: {delta:+.1f}%.")
        lines.append("")

        lines += [
            "### Spatial Mask",
            "",
            f"- Valid pixels: **{r['n_valid']:,}** / {r['n_total']:,}",
            f"- Pixel-to-band ratio: {r['px_band_ratio']}",
            "",
            "### RX Score Distribution",
            "",
            "| p2 | p25 | median | p75 | p98 | p99 | p99.9 | max |",
            "|---|---|---|---|---|---|---|---|",
        ]
        p = r["percs"]
        lines.append(
            f"| {p['p2']:.1f} | {p['p25']:.1f} | {p['median']:.1f} | "
            f"{p['p75']:.1f} | {p['p98']:.1f} | {p['p99']:.1f} | "
            f"{p['p999']:.1f} | {p['max']:.0f} |"
        )
        lines += [
            "",
            f"- Outlier pixels above p99: **{r['n_outliers_p99']:,}**",
            f"- Outlier pixels above p99.9: **{r['n_outliers_p999']:,}**",
            "",
            "### Per-Band Stripe Reduction",
            "",
            "| Band | Wavelength | Label | σ before | σ after | Δ |",
            "|---|---|---|---|---|---|",
        ]
        for s in r["stripe"]:
            direction = "↓" if s["delta_pct"] < 0 else "↑"
            lines.append(
                f"| {s['band']} | {s['wavelength']:.1f}nm | {s['label']} | "
                f"{s['sigma_before']:.6f} | {s['sigma_after']:.6f} | "
                f"{direction}{abs(s['delta_pct']):.1f}% |"
            )
        lines += [
            "",
            f"![{r['scene_id']}]({r['fig_name']})",
            "",
            "---",
            "",
        ]

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
            results.append(run_single(path))
        except Exception as e:
            logger.error("FAILED on %s: %s", path.name, e, exc_info=True)

    if results:
        write_summary(results)

    logger.info("Done. %d/%d scenes.", len(results), len(he5_files))


if __name__ == "__main__":
    main()
