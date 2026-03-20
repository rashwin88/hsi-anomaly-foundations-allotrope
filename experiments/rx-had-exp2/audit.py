"""
Pipeline audit: destripe + RX end-to-end for a single scene.
Reports each step without modifying anything.
"""

import sys
import logging
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Silence all library loggers so only our output appears
logging.basicConfig(level=logging.WARNING)

from app.models.file_processing.sources import FileSourceConfig
from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
from app.utils.data_transformations.spectral_band_filter import (
    SpectralBandFilter, DEFAULT_EXCLUSION_RANGES,
)
from app.utils.data_transformations.frequency_domain_destriper import (
    FrequencyDomainDestriper, ANGLE_SEARCH_STEP, PEAK_SIGMA, RADIAL_SKIP,
)
from app.utils.data_transformations.moment_matching_destriper import MomentMatchingDestriper
from app.utils.data_transformations.composite_destriper import CombinedDestriper
from app.detectors.global_rx_detector import GlobalRXDetector

HE5 = (
    PROJECT_ROOT
    / "tests/test_payloads/phase_2/Set-4/Hyper"
    / "PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
)

SEP = "=" * 70


def col_mean_sigma(band: np.ndarray, validity_2d: np.ndarray) -> float:
    """std of per-column means over valid pixels."""
    col_means = []
    for j in range(band.shape[1]):
        mask = validity_2d[:, j].astype(bool)
        if mask.sum() > 0:
            col_means.append(band[mask, j].mean())
    return float(np.std(col_means)) if col_means else 0.0


# ── 0. Build vendable ─────────────────────────────────────────────────────────
print(SEP)
print("BUILDING VENDABLE")
print(SEP)
builder = PrismaDatasetBuilder(
    file_source_configuration=FileSourceConfig(source_path=str(HE5))
)
vendable = builder.vend_dataset()

cube     = vendable.normalized_hyperspectral_cube   # (B, H, W)
validity = vendable.validity_cube                   # (B, H, W)
wavelengths = vendable.band_cw_order                # List[float]
band_flags  = vendable.band_validity_by_position    # List[int]

B, H, W = cube.shape
print(f"Cube shape  : {cube.shape}")
print(f"Wavelengths : {len(wavelengths)} entries, "
      f"{min(wavelengths):.1f}–{max(wavelengths):.1f} nm")
print(f"Band flags  : {sum(1 for f in band_flags if f == 1)} valid / {len(band_flags)} total")


# ── 1. BAND FILTERING ─────────────────────────────────────────────────────────
print()
print(SEP)
print("BAND FILTERING SUMMARY")
print(SEP)

# Stage 1a: validity flags
dropped_flags = [i for i, f in enumerate(band_flags) if f != 1]
after_flags   = [i for i, f in enumerate(band_flags) if f == 1]
print(f"Input            : {B} bands")
print(f"Stage 1a (flags) : dropped {len(dropped_flags)}, surviving {len(after_flags)}")
if dropped_flags:
    print(f"  Dropped indices: {dropped_flags}")

# Stage 1b: wavelength exclusion
sbf = SpectralBandFilter(wavelengths, band_flags, DEFAULT_EXCLUSION_RANGES)
sbf_summary = sbf.summary()
good_indices_stage1b = sbf.get_good_band_indices()

dropped_wl = sbf_summary["dropped_by_wavelength"]
print(f"Stage 1b (wl)    : dropped {dropped_wl['count']}, "
      f"surviving {sbf_summary['surviving']}")

# Which exclusion range caught each dropped-by-wl band?
range_label = {
    (0, 450):       "low-SNR / detector noise",
    (912, 978):     "water vapour 912–978nm",
    (1131, 1152):   "water vapour 1131–1152nm",
    (1328, 1492):   "water vapour deep 1328–1492nm",
    (1784, 1967):   "water vapour + CO₂ 1784–1967nm",
}
for idx, rng in dropped_wl["ranges"].items():
    label = range_label.get(rng, str(rng))
    print(f"  Band {idx:3d} ({wavelengths[idx]:.1f}nm) → {label}")

# Stage 2: per-pixel failure rate (replicate detector logic)
FAIL_THRESH = 0.05
any_valid    = validity.any(axis=0)
n_spatial    = int(any_valid.sum())
good_after_stage2 = []
dropped_stage2    = []
for i in good_indices_stage1b:
    n_invalid = n_spatial - int(validity[i][any_valid].sum())
    fail_rate = n_invalid / n_spatial
    if fail_rate > FAIL_THRESH:
        dropped_stage2.append((i, fail_rate))
    else:
        good_after_stage2.append(i)

print(f"Stage 2 (px fail): dropped {len(dropped_stage2)}, "
      f"surviving {len(good_after_stage2)}")
for idx, rate in dropped_stage2:
    print(f"  Band {idx:3d} ({wavelengths[idx]:.1f}nm) — {rate*100:.1f}% pixel failure")

print(f"Final            : {len(good_after_stage2)} bands")


# ── 2. FFT DESTRIPER ─────────────────────────────────────────────────────────
print()
print(SEP)
print("FFT DESTRIPER SUMMARY")
print(SEP)

from scipy.ndimage import gaussian_filter

fft_d = FrequencyDomainDestriper()

# Run with diagnostics=True to capture everything
destriped_cube = fft_d.transform(
    cube,
    validity_mask=validity,
    diagnostics=True,
    band_wavelengths=wavelengths,
)
fft_diag = fft_d.diagnostics

# Probe bands (re-run selection for reporting with wavelengths)
probe_indices = fft_d._select_probe_bands(
    cube, validity,
    band_wavelengths=wavelengths,
)
print(f"Probe bands ({len(probe_indices)}):")
for idx in probe_indices:
    print(f"  Band {idx:3d} — {wavelengths[idx]:.1f} nm")

# Per-probe peak angle and strength
print("\nPer-probe peak analysis:")
angles_grid = np.arange(0, 180, ANGLE_SEARCH_STEP)
for idx in probe_indices:
    band  = cube[idx]
    valid = validity[idx].astype(bool)
    if not valid.any():
        print(f"  Band {idx:3d}: no valid pixels — skipped")
        continue
    bm     = band[valid].mean()
    filled = np.where(valid, band, bm)
    fft_s  = np.fft.fftshift(np.fft.fft2(filled))
    lp     = np.log(np.abs(fft_s) ** 2 + 1e-10)
    _, sums = fft_d._radial_power_vs_angle(lp)

    pk_i   = int(np.argmax(sums))
    pk_ang = angles_grid[pk_i]
    adiff  = np.abs(angles_grid - pk_ang)
    adiff  = np.minimum(adiff, 180 - adiff)
    bg     = sums[adiff > 10.0]
    bg_m, bg_s = bg.mean(), bg.std()
    strength = (sums[pk_i] - bg_m) / bg_s if bg_s > 0 else 0.0
    sig = "✓ significant" if strength > PEAK_SIGMA else "✗ below threshold"
    print(f"  Band {idx:3d} ({wavelengths[idx]:.1f}nm): peak={pk_ang:.1f}°  "
          f"strength={strength:.1f}σ  {sig}")

# Consensus result
detected = fft_diag.detected_angles or []
strengths = fft_diag.angle_strengths or []
rps       = fft_diag.angle_radial_preserves or []
print(f"\nConsensus angles: {len(detected)}")
for ang, stre, rp in zip(detected, strengths, rps):
    print(f"  {ang:.1f}° — strength={stre:.1f}σ, radial_preserve={rp}")

from app.utils.data_transformations.frequency_domain_destriper import PAD_WIDTH
print(f"Pad width: {PAD_WIDTH} px")


# ── 3. MOMENT MATCHING ───────────────────────────────────────────────────────
print()
print(SEP)
print("MOMENT MATCHING SUMMARY")
print(SEP)

# CombinedDestriper reads detected_angles from fft_diag, but only when diagnostics=True.
# When diagnostics=False that path is broken — report this explicitly.
if not detected:
    print("⚠️  No angles passed to moment matcher.")
    print("   (This is the bug: fft._diagnostics is None when diagnostics=False,")
    print("    so CombinedDestriper always skips moment-matching in production.)")
    mm_cubes = {}
else:
    print(f"Angles passed to moment matcher: {[f'{a:.1f}°' for a in detected]}")

    rep = fft_diag.representative_band_index
    rep_validity_2d = validity[rep].astype(bool)

    sigma_orig = col_mean_sigma(cube[rep], rep_validity_2d)
    sigma_fft  = col_mean_sigma(destriped_cube[rep], rep_validity_2d)
    print(f"\nColumn-mean σ (rep band {rep}, {wavelengths[rep]:.1f}nm):")
    print(f"  Original : {sigma_orig:.6f}")
    print(f"  After FFT: {sigma_fft:.6f}")

    mm_d   = MomentMatchingDestriper()
    result = destriped_cube.copy()
    for angle in detected:
        candidate = mm_d.transform(result, validity_mask=validity, stripe_angle=angle)
        s_before  = col_mean_sigma(result[rep],    rep_validity_2d)
        s_after   = col_mean_sigma(candidate[rep], rep_validity_2d)
        guard_fired = s_after > s_before
        print(f"\n  MM pass at {angle:.1f}°:")
        print(f"    σ before = {s_before:.6f}")
        print(f"    σ after  = {s_after:.6f}")
        if guard_fired:
            print(f"    σ guard  : TRIGGERED — would skip this angle")
        else:
            print(f"    σ guard  : OK — applying")
            result = candidate
        mm_cubes = result

    sigma_mm = col_mean_sigma(result[rep], rep_validity_2d)
    print(f"\nColumn-mean σ progression:")
    print(f"  original={sigma_orig:.6f} → FFT={sigma_fft:.6f} → MM={sigma_mm:.6f}")

    # Use the properly moment-matched result for RX below
    destriped_cube = result


# ── 4. SPATIAL MASK ──────────────────────────────────────────────────────────
print()
print(SEP)
print("SPATIAL MASK SUMMARY")
print(SEP)

# Replicate the spatial mask logic from GlobalRXDetector
good_bands = good_after_stage2
spatial_mask = np.ones((H, W), dtype=bool)
for i in good_bands:
    spatial_mask &= validity[i].astype(bool)

n_valid = int(spatial_mask.sum())
n_total = H * W
pb_ratio = n_valid / len(good_bands) if good_bands else 0
print(f"Valid pixels     : {n_valid:,} / {n_total:,}")
print(f"Pixel-to-band    : {pb_ratio:.1f}")

# Check for non-finite values in the destriped cube at valid pixels
finite_check = True
for i in good_bands:
    vals = destriped_cube[i][spatial_mask]
    if not np.all(np.isfinite(vals)):
        finite_check = False
        n_bad = int(np.sum(~np.isfinite(vals)))
        print(f"⚠️  Band {i} has {n_bad} non-finite values at valid pixels")
if finite_check:
    print("All finite       : YES")


# ── 5. RX COMPUTATION ────────────────────────────────────────────────────────
print()
print(SEP)
print("RX SUMMARY")
print(SEP)

detector = GlobalRXDetector(vendable)
detector.fit(band_failure_threshold=FAIL_THRESH)
score_map = detector.detect(destriped_cube, validity)
result    = detector.result

valid_scores = score_map[~np.isnan(score_map)]
print(f"Score stats (valid pixels only):")
print(f"  min    = {valid_scores.min():.4f}")
print(f"  max    = {valid_scores.max():.4f}")
print(f"  mean   = {valid_scores.mean():.4f}")
print(f"  median = {float(np.median(valid_scores)):.4f}")
print(f"  p2     = {float(np.percentile(valid_scores, 2)):.4f}")
print(f"  p98    = {float(np.percentile(valid_scores, 98)):.4f}")
print(f"  p99    = {float(np.percentile(valid_scores, 99)):.4f}")
print(f"  p99.9  = {float(np.percentile(valid_scores, 99.9)):.4f}")


# ── 6. PER-BAND STRIPE CHECK ─────────────────────────────────────────────────
print()
print(SEP)
print("PER-BAND STRIPE CHECK")
print(SEP)

# Pick 3 representative bands
def pick_band_near(target_nm: float, candidates: list[int]) -> int:
    return min(candidates, key=lambda i: abs(wavelengths[i] - target_nm))

b_vnir     = pick_band_near(600,  good_after_stage2)
b_boundary = pick_band_near(1000, good_after_stage2)
b_swir     = pick_band_near(2000, good_after_stage2)

family_map = {i: str(vendable.spectral_family_order[i]) for i in [b_vnir, b_boundary, b_swir]}

for band_idx, label in [(b_vnir, "VNIR ~600nm"),
                         (b_boundary, "boundary ~1000nm"),
                         (b_swir, "SWIR ~2000nm")]:
    v2d = validity[band_idx].astype(bool)
    s_before = col_mean_sigma(cube[band_idx],          v2d)
    s_after  = col_mean_sigma(destriped_cube[band_idx], v2d)
    pct      = (s_after - s_before) / s_before * 100 if s_before > 0 else 0
    direction = "↓ reduced" if s_after < s_before else "↑ INCREASED"
    print(f"Band {band_idx:3d} ({wavelengths[band_idx]:.1f}nm, {label}):")
    print(f"  σ before={s_before:.6f}  σ after={s_after:.6f}  "
          f"Δ={pct:+.1f}%  {direction}")


# ── DIAGNOSIS ────────────────────────────────────────────────────────────────
print()
print(SEP)
print("DIAGNOSIS")
print(SEP)

if not detected:
    print("CRITICAL BUG: Moment-matching was never applied because CombinedDestriper")
    print("reads detected_angles from self._fft_destriper._diagnostics, which is None")
    print("when diagnostics=False. The FFT notch ran, but the MM pass was silently")
    print("skipped for every scene in production mode. This means stripes that the")
    print("FFT notch cannot fully suppress (low-frequency gain variation) persist")
    print("in the destriped cube and pollute the RX score map.")
else:
    print("Moment-matching was applied. See σ progression above.")
