"""
Automatic frequency-domain destriper for hyperspectral cubes in BSQ format.

Removes stripe noise from pushbroom sensor data by detecting the stripe
angle in the frequency domain and applying a directional notch filter.
Works for any sensor and any stripe angle.

Three stages:
    1. Find the stripe angle from multiple bands' power spectra (consensus).
    2. Build a tapered notch filter at that angle.
    3. Apply the filter to every band (batched via torch.fft on the best
       available device: CUDA > MPS > CPU).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation
from app.utils.torch_helpers.device_selection import get_device

logger = logging.getLogger(__name__)

DEFAULT_ANGULAR_WIDTH = 3.0
DEFAULT_RADIAL_PRESERVE = 5
DEFAULT_TAPER_SIGMA = 2.0
PEAK_SIGMA = 3.0
ANGLE_SEARCH_STEP = 0.5
RADIAL_SKIP = 10
ANGLE_TOLERANCE = 3.0  # degrees — max spread for consensus across bands
PAD_WIDTH = 128
# Half-width of the window excluded around a candidate angle when measuring the
# background it is scored against. Wide enough to exclude the peak's own
# shoulders, narrow enough to leave most of the 180-degree curve as background.
BACKGROUND_EXCLUSION_DEG = 10.0


def _background_stats(
    sums: np.ndarray, angles_grid: np.ndarray, angle: float
) -> tuple[float, float]:
    """
    Mean and standard deviation of the power-vs-angle curve, ignoring the
    neighbourhood of `angle`.

    A stripe shows up as a narrow peak in that curve, so "is this peak real?"
    means "does it stand out from the rest of the curve". The +/- window is
    excluded so the peak does not inflate the background it is measured
    against.

    Note the std can legitimately be zero: a perfectly clean synthetic scene
    has a flat curve away from the peak. Callers must guard, or every stripe in
    a noiseless image is silently rejected - see the destriper's entry in
    docs/09-known-issues.md.
    """
    diff = np.abs(angles_grid - angle)
    diff = np.minimum(diff, 180 - diff)
    bg = sums[diff > BACKGROUND_EXCLUSION_DEG]
    return bg.mean(), bg.std()


def _band_means(cube: np.ndarray, validity_bool: np.ndarray) -> np.ndarray:
    """
    Mean of the valid pixels in each band, as float32 (B,).

    Used as the fill value for invalid pixels and as the pad value around the
    frame. Filling with the band's own mean rather than zero matters: a zero
    against a reflectance scene is a huge step edge, and a step edge is
    broadband in the frequency domain, so it would smear energy across every
    angle and swamp the stripe peak the notch is trying to find.
    """
    return np.array(
        [
            cube[b][validity_bool[b]].mean() if validity_bool[b].any() else 0.0
            for b in range(cube.shape[0])
        ],
        dtype=np.float32,
    )


def _choose_batch_size(
    n_bands: int, padded_h: int, padded_w: int, device: torch.device
) -> int:
    """
    How many bands can go through the FFT at once without exhausting memory.

    Each in-flight band costs a padded float32 input, a complex64 spectrum, a
    complex64 filtered copy and a float32 result - 24 bytes per padded pixel.
    A padded PRISMA band is ~1450x1500, so ~52 MB each; 239 of them at once
    would be 12 GB.
    """
    bytes_per_band = padded_h * padded_w * (4 + 8 + 8 + 4)
    mem_budget = 2e9 if device.type == "cuda" else 500e6
    batch_size = min(n_bands, max(1, int(mem_budget / bytes_per_band)))
    logger.info(
        "FFT filter: batch_size=%d (%.0f MB per band)",
        batch_size, bytes_per_band / 1e6,
    )
    return batch_size


def _fill_and_pad(
    cube_chunk: np.ndarray,
    validity_chunk: np.ndarray,
    means: np.ndarray,
    pad: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Move a batch to the device, fill invalid pixels, and pad the frame.

    Both fill and pad use the band mean, for the reason given in _band_means.
    Padding at all is what stops the FFT treating opposite edges of the scene
    as adjacent, which would inject a spurious edge at every wrap point.
    """
    chunk = torch.from_numpy(cube_chunk.astype(np.float32)).to(device)
    valid = torch.from_numpy(validity_chunk).to(device)
    mean_t = torch.from_numpy(means).to(device)

    chunk = torch.where(valid, chunk, mean_t[:, None, None].expand_as(chunk))

    padded = torch.nn.functional.pad(
        chunk, (pad, pad, pad, pad), mode="constant", value=0
    )
    # F.pad cannot take a per-band constant, so the border is overwritten
    # afterwards. Rows first, then columns - the corners end up as columns.
    padded[:, :pad, :] = mean_t[:, None, None]
    padded[:, -pad:, :] = mean_t[:, None, None]
    padded[:, :, :pad] = mean_t[:, None, None]
    padded[:, :, -pad:] = mean_t[:, None, None]
    return padded


def _notch_batch(
    padded: torch.Tensor, notch_t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    FFT -> multiply by the notch -> inverse FFT.

    Returns (spatial_result, filtered_spectrum). The spectrum comes back only
    so the caller can capture it for diagnostics; it is otherwise dead weight.

    fftshift moves the zero frequency to the centre, which is what makes the
    notch a simple angular wedge through the middle rather than something that
    has to wrap around four corners.
    """
    spectrum = torch.fft.fftshift(torch.fft.fft2(padded), dim=(-2, -1))
    filtered = spectrum * notch_t
    result = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1))).real
    return result, filtered


@dataclass
class FrequencyDestripeDiagnostics:
    """Optional diagnostics captured when diagnostics=True."""

    angles: np.ndarray = field(default=None)
    power_sums: np.ndarray = field(default=None)
    detected_angle: float = field(default=None)       # primary angle (for display)
    detected_angles: list = field(default=None)       # all consensus angles
    significance_threshold: float = field(default=None)
    log_power_before: np.ndarray = field(default=None)
    log_power_after: np.ndarray = field(default=None)
    notch_filter: np.ndarray = field(default=None)
    band_before: np.ndarray = field(default=None)
    band_after: np.ndarray = field(default=None)
    representative_band_index: int = field(default=None)
    angle_strengths: list = field(default=None)          # σ above bg per angle
    angle_radial_preserves: list = field(default=None)   # chosen rp per angle


class FrequencyDomainDestriper(DataTransformer):
    """
    Automatic frequency-domain destriper.

    Call transform(cube, validity_mask=mask) to destripe.
    Pass diagnostics=True to capture intermediate results.
    """

    def __init__(
        self,
        angular_width: float = DEFAULT_ANGULAR_WIDTH,
        radial_preserve: int | None = None,
        taper_sigma: float = DEFAULT_TAPER_SIGMA,
    ):
        super().__init__(
            transformation_category=Transformation.FREQUENCY_DOMAIN_DESTRIPE
        )
        self._angular_width = angular_width
        self._radial_preserve = radial_preserve  # None = adaptive
        self._taper_sigma = taper_sigma
        self._diagnostics: FrequencyDestripeDiagnostics | None = None

    @property
    def diagnostics(self) -> FrequencyDestripeDiagnostics:
        if self._diagnostics is None:
            raise RuntimeError(
                "No diagnostics available. Call transform() with diagnostics=True."
            )
        return self._diagnostics

    def transform(self, input_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Destripe a BSQ reflectance cube.

        Args:
            input_data:     (B, H, W) reflectance cube.
            validity_mask:  (B, H, W) binary mask, 1 = valid. Required.
            diagnostics:    bool, default False. Capture intermediate results.

        Returns:
            Destriped cube, same shape and dtype as input_data.
        """
        validity_mask = kwargs.get("validity_mask")
        if validity_mask is None:
            raise ValueError("validity_mask is required.")

        capture = kwargs.get("diagnostics", False)
        band_wavelengths      = kwargs.get("band_wavelengths", None)
        exclusion_ranges      = kwargs.get("exclusion_ranges", None)
        spectral_family_order = kwargs.get("spectral_family_order", None)

        diag = FrequencyDestripeDiagnostics() if capture else None
        self._diagnostics = diag
        self._last_detected_angles: list[float] = []  # always populated

        original_dtype = input_data.dtype
        B, H, W = input_data.shape
        logger.info("FFT destriper: cube (%d, %d, %d), dtype=%s", B, H, W, original_dtype)

        # Stage 1: find all stripe angles with strength
        t0 = time.time()
        angle_strength_pairs = self._find_stripe_angles(
            input_data, validity_mask, diag,
            band_wavelengths=band_wavelengths,
            exclusion_ranges=exclusion_ranges,
            spectral_family_order=spectral_family_order,
        )
        logger.info("FFT destriper: angle detection done in %.2fs", time.time() - t0)

        if not angle_strength_pairs:
            logger.warning("No significant stripes detected. Returning cube unchanged.")
            return input_data.copy()

        logger.info(
            "Detected %d stripe angle(s): %s",
            len(angle_strength_pairs),
            [f"{a:.1f}° ({s:.1f}σ)" for a, s in angle_strength_pairs],
        )

        # Stage 2: build per-angle notch filters with adaptive radial preserve,
        # combine by element-wise minimum
        padded_shape = (H + 2 * PAD_WIDTH, W + 2 * PAD_WIDTH)
        notch = np.ones(padded_shape, dtype=np.float64)
        for angle, strength in angle_strength_pairs:
            rp = (
                self._radial_preserve
                if self._radial_preserve is not None
                else self._strength_to_radial_preserve(strength)
            )
            logger.info(
                "Notch at %.1f°: strength=%.1fσ, radial_preserve=%d", angle, strength, rp
            )
            per_angle_notch = self._build_notch_filter(padded_shape, angle, rp)
            notch = np.minimum(notch, per_angle_notch)

        angles_only = [a for a, _ in angle_strength_pairs]
        self._last_detected_angles = angles_only  # always stored
        if diag is not None:
            diag.notch_filter = notch
            diag.detected_angles = angles_only
            diag.detected_angle = angles_only[0]

        # Stage 3: apply to every band (batched on device)
        t0 = time.time()
        output = self._apply_filter(input_data, validity_mask, notch, diag)
        logger.info("FFT destriper: applied notch to %d bands in %.2fs", B, time.time() - t0)

        return output.astype(original_dtype)

    def _find_stripe_angles(
        self,
        cube: np.ndarray,
        validity: np.ndarray,
        diag: FrequencyDestripeDiagnostics | None,
        band_wavelengths: list[float] | None = None,
        exclusion_ranges: list | None = None,
        spectral_family_order: list | None = None,
    ) -> list[Tuple[float, float]]:
        """
        Detect all stripe angles present in the scene via multi-band
        consensus. Returns a list of (angle, strength) tuples (may be empty).
        Strength is σ above local background in the power-vs-angle curve.
        """
        probe_indices = self._select_probe_bands(
            cube, validity,
            band_wavelengths=band_wavelengths,
            exclusion_ranges=exclusion_ranges,
            spectral_family_order=spectral_family_order,
        )
        logger.info("Probing %d bands for stripe angles: %s", len(probe_indices), probe_indices)

        angles_grid = np.arange(0, 180, ANGLE_SEARCH_STEP)
        candidate_angles: list[float] = []

        for idx in probe_indices:
            band = cube[idx]
            valid = validity[idx].astype(bool)
            if not valid.any():
                continue
            band_mean = band[valid].mean()
            filled = self._fill_invalid(band, valid, band_mean)
            log_power = self._compute_log_power_spectrum(filled)
            _, sums = self._radial_power_vs_angle(log_power)

            peak_idx = np.argmax(sums)
            peak_angle = angles_grid[peak_idx]
            bg_mean, bg_std = _background_stats(sums, angles_grid, peak_angle)

            if bg_std > 0 and sums[peak_idx] > bg_mean + PEAK_SIGMA * bg_std:
                candidate_angles.append(peak_angle)

        if len(candidate_angles) == 0:
            logger.warning("No band showed a significant peak. No stripes found.")
            return []

        consensus_angles = self._find_all_consensus_angles(candidate_angles)

        if not consensus_angles:
            logger.warning(
                "No consistent angle across bands. Peaks at: %s. "
                "Likely scene content, not stripes.",
                [f"{a:.1f}°" for a in candidate_angles],
            )
            return []

        # Compute stripe strength per consensus angle using the representative
        # band's power curve
        rep_idx = probe_indices[len(probe_indices) // 2]
        valid = validity[rep_idx].astype(bool)
        band_mean = cube[rep_idx][valid].mean()
        filled = self._fill_invalid(cube[rep_idx], valid, band_mean)
        log_power = self._compute_log_power_spectrum(filled)
        _, rep_sums = self._radial_power_vs_angle(log_power)

        angle_strength_pairs: list[Tuple[float, float]] = []
        for angle in consensus_angles:
            # Find the power sum at this angle
            angle_idx = int(round(angle / ANGLE_SEARCH_STEP))
            angle_idx = min(angle_idx, len(rep_sums) - 1)
            peak_sum = rep_sums[angle_idx]

            bg_mean, bg_std = _background_stats(rep_sums, angles_grid, angle)
            strength = (peak_sum - bg_mean) / bg_std if bg_std > 0 else 0.0

            n_agree = sum(
                1 for a in candidate_angles
                if min(abs(a - angle), 180 - abs(a - angle)) <= ANGLE_TOLERANCE
            )
            logger.info(
                "Consensus stripe angle: %.1f° (%d/%d probe bands agree, "
                "strength=%.1fσ)",
                angle, n_agree, len(probe_indices), strength,
            )
            angle_strength_pairs.append((angle, strength))

        # Capture diagnostics
        if diag is not None:
            peak_idx = np.argmax(rep_sums)
            angle_diff = np.abs(angles_grid - angles_grid[peak_idx])
            angle_diff = np.minimum(angle_diff, 180 - angle_diff)
            bg = rep_sums[angle_diff > 10.0]
            threshold = bg.mean() + PEAK_SIGMA * bg.std()

            diag.angles = angles_grid
            diag.power_sums = rep_sums
            diag.significance_threshold = threshold
            diag.log_power_before = log_power
            diag.representative_band_index = rep_idx
            diag.angle_strengths = [s for _, s in angle_strength_pairs]
            diag.angle_radial_preserves = [
                self._strength_to_radial_preserve(s) for _, s in angle_strength_pairs
            ]

        return angle_strength_pairs

    @staticmethod
    def _strength_to_radial_preserve(strength: float) -> int:
        """Map stripe strength (σ above background) to radial preserve radius."""
        if strength > 10:
            return 2
        elif strength > 5:
            return 3
        else:
            return 5

    @staticmethod
    def _select_probe_bands(
        cube: np.ndarray,
        validity: np.ndarray,
        n_probes_per_family: int = 10,
        band_wavelengths: list[float] | None = None,
        exclusion_ranges: list | None = None,
        spectral_family_order: list | None = None,
    ) -> list[int]:
        """
        Select up to n_probes_per_family bands from each spectral family
        (VNIR and SWIR), for a maximum of 2 * n_probes_per_family probes.

        When spectral_family_order is not provided, falls back to picking
        n_probes_per_family bands from the full spectral range (original
        behaviour).

        Selection per family:
          1. Restrict to bands belonging to that family.
          2. Apply wavelength exclusion (SpectralBandFilter).
          3. Keep only bands with < 10% pixel failure.
          4. Pick n_probes_per_family evenly spaced across wavelength range.
          5. Among candidates, prefer highest variance (same as before).
        """
        from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

        B = validity.shape[0]
        any_valid = validity.any(axis=0)
        n_spatial = int(any_valid.sum())

        invalid_counts = np.array([
            n_spatial - validity[b][any_valid].sum() for b in range(B)
        ])

        # Spectral-filter good set (wavelength exclusion + flags)
        if band_wavelengths is not None:
            from app.utils.data_transformations.spectral_band_filter import SpectralBandFilter
            spectral_good = set(
                SpectralBandFilter(
                    band_wavelengths=band_wavelengths,
                    band_validity_flags=[1] * B,
                    exclusion_ranges=exclusion_ranges,
                ).get_good_band_indices()
            )
        else:
            spectral_good = set(range(B))

        def _pick_from_pool(pool: list[int]) -> list[int]:
            """Pick n_probes_per_family from a pool, evenly spaced, prefer high variance."""
            if not pool:
                return []
            arr = np.array(pool)
            step = max(1, len(arr) // n_probes_per_family)
            candidates = arr[::step][:n_probes_per_family]
            variances = []
            for idx in candidates:
                v = validity[idx].astype(bool)
                variances.append(cube[idx][v].var() if v.any() else 0.0)
            order = np.argsort(variances)[::-1]
            return [int(candidates[i]) for i in order[:n_probes_per_family]]

        if spectral_family_order is None:
            # Original single-pool behaviour
            usable = [
                b for b in range(B)
                if invalid_counts[b] < 0.10 * n_spatial and b in spectral_good
            ]
            if not usable:
                usable = list(range(B))
            return _pick_from_pool(usable)

        # Partitioned selection: one pool per family
        probes: list[int] = []
        for family in (SpectralFamily.VNIR, SpectralFamily.SWIR):
            pool = [
                b for b in range(B)
                if (
                    spectral_family_order[b] == family
                    and b in spectral_good
                    and invalid_counts[b] < 0.10 * n_spatial
                )
            ]
            if not pool:
                logger.warning(
                    "No usable probe bands for family %s — skipping.", family.value
                )
                continue
            picked = _pick_from_pool(pool)
            logger.info(
                "Probe bands for %s (%d): %s", family.value, len(picked), picked
            )
            probes.extend(picked)

        if not probes:
            # Both families failed — fall back to full-range
            logger.warning("No family probes available, falling back to full-range selection.")
            usable = [b for b in range(B) if invalid_counts[b] < 0.10 * n_spatial]
            probes = _pick_from_pool(usable or list(range(B)))

        return probes

    @staticmethod
    def _find_all_consensus_angles(
        candidate_angles: list[float],
        min_support: int = 2,
    ) -> list[float]:
        """
        Find all angle clusters with sufficient support.

        Greedily extracts clusters: find the angle with most agreement,
        record it, remove those candidates, repeat until no cluster has
        enough support.

        Returns list of consensus angles (may be empty).
        """
        if len(candidate_angles) == 0:
            return []

        remaining = list(candidate_angles)
        consensus_angles: list[float] = []
        required = max(min_support, len(candidate_angles) // 4)

        while remaining:
            # Find the angle with most agreement among remaining
            best_ref = None
            best_count = 0
            for ref in remaining:
                count = sum(
                    1 for a in remaining
                    if min(abs(a - ref), 180 - abs(a - ref)) <= ANGLE_TOLERANCE
                )
                if count > best_count:
                    best_count = count
                    best_ref = ref

            if best_count < required:
                break

            # Extract this cluster
            agreeing = [
                a for a in remaining
                if min(abs(a - best_ref), 180 - abs(a - best_ref)) <= ANGLE_TOLERANCE
            ]
            consensus_angles.append(float(np.mean(agreeing)))

            # Remove used candidates
            remaining = [
                a for a in remaining
                if min(abs(a - best_ref), 180 - abs(a - best_ref)) > ANGLE_TOLERANCE
            ]

        return consensus_angles

    @staticmethod
    def _select_representative_band(
        cube: np.ndarray, validity: np.ndarray
    ) -> int:
        """
        Pick the band with fewest invalid pixels. Among ties (within 1%
        of each other), prefer the band with highest variance across valid
        pixels — a high-variance band gives a stronger stripe-to-noise
        ratio in the power spectrum.
        """
        B = validity.shape[0]
        # Use swath-valid pixel count, not total H*W
        any_valid = validity.any(axis=0)
        n_spatial = int(any_valid.sum())
        invalid_counts = np.array([
            n_spatial - validity[b][any_valid].sum() for b in range(B)
        ])

        min_invalid = invalid_counts.min()
        tolerance = 0.01 * n_spatial
        candidates = np.where(invalid_counts <= min_invalid + tolerance)[0]

        if len(candidates) == 1:
            return int(candidates[0])

        # Among candidates, pick highest variance across valid pixels
        best_idx = candidates[0]
        best_var = -1.0
        for idx in candidates:
            valid = validity[idx].astype(bool)
            if valid.any():
                var = cube[idx][valid].var()
                if var > best_var:
                    best_var = var
                    best_idx = idx

        return int(best_idx)

    @staticmethod
    def _fill_invalid(
        band: np.ndarray, valid_mask: np.ndarray, band_mean: float
    ) -> np.ndarray:
        """Fill invalid pixels with the precomputed band mean."""
        filled = band.copy()
        filled[~valid_mask] = band_mean
        return filled

    @staticmethod
    def _compute_log_power_spectrum(filled_band: np.ndarray) -> np.ndarray:
        fft = np.fft.fft2(filled_band)
        fft_shifted = np.fft.fftshift(fft)
        power = np.abs(fft_shifted) ** 2
        return np.log(power + 1e-10)

    @staticmethod
    def _radial_power_vs_angle(
        log_power: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        For angles 0–179 in half-degree steps, sum power along radial lines
        from the center, skipping the inner RADIAL_SKIP pixels.

        Fully vectorized: precomputes all (angle, radius) sample coordinates
        in one shot, then gathers and sums via reshape.
        """
        H, W = log_power.shape
        cy, cx = H // 2, W // 2
        max_radius = min(cy, cx)

        angles = np.arange(0, 180, ANGLE_SEARCH_STEP)
        n_angles = len(angles)
        radii = np.arange(RADIAL_SKIP, max_radius)
        n_radii = len(radii)

        # (n_angles,) and (n_radii,) → broadcast to (n_angles, n_radii)
        angles_rad = np.deg2rad(angles)
        sin_a = np.sin(angles_rad)[:, np.newaxis]   # (n_angles, 1)
        cos_a = np.cos(angles_rad)[:, np.newaxis]   # (n_angles, 1)
        r = radii[np.newaxis, :]                     # (1, n_radii)

        # Both directions from center → (n_angles, 2 * n_radii)
        rows = np.concatenate([
            cy + (r * sin_a).astype(int),
            cy - (r * sin_a).astype(int),
        ], axis=1)
        cols = np.concatenate([
            cx + (r * cos_a).astype(int),
            cx - (r * cos_a).astype(int),
        ], axis=1)

        # Clamp to valid bounds
        valid = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
        rows_safe = np.clip(rows, 0, H - 1)
        cols_safe = np.clip(cols, 0, W - 1)

        # Gather all values, zero out-of-bounds, sum per angle
        values = log_power[rows_safe, cols_safe]
        values[~valid] = 0.0
        sums = values.sum(axis=1)

        return angles, sums

    # ------------------------------------------------------------------
    # Stage 2 — Build notch filter
    # ------------------------------------------------------------------

    def _build_notch_filter(
        self, shape: Tuple[int, int], stripe_angle: float, radial_preserve: int = 5
    ) -> np.ndarray:
        H, W = shape
        cy, cx = H // 2, W // 2

        ys = np.arange(H) - cy
        xs = np.arange(W) - cx
        xx, yy = np.meshgrid(xs, ys)
        pixel_angles = np.rad2deg(np.arctan2(yy, xx)) % 180

        dist = np.sqrt(xx ** 2 + yy ** 2)

        stripe_mod = stripe_angle % 180
        angle_diff = np.abs(pixel_angles - stripe_mod)
        angle_diff = np.minimum(angle_diff, 180 - angle_diff)

        notch = np.ones((H, W), dtype=np.float64)
        notch[angle_diff <= self._angular_width] = 0.0

        # Smooth the wedge edges first
        notch = gaussian_filter(notch, sigma=self._taper_sigma)

        # Then force the center back to 1.0 — hard protection, not blurred
        notch[dist <= radial_preserve] = 1.0

        return notch

    # ------------------------------------------------------------------
    # Stage 3 — Apply filter to every band (batched via torch.fft)
    # ------------------------------------------------------------------

    def _apply_filter(
        self,
        cube: np.ndarray,
        validity: np.ndarray,
        notch: np.ndarray,
        diag: FrequencyDestripeDiagnostics | None,
    ) -> np.ndarray:
        """
        Apply the notch filter to every band, in device-sized batches.

        Each batch is prepare -> filter -> write back. Everything except the
        output array stays on-device, so a 239-band cube never materialises a
        second full-size float array on the host.
        """
        B, H, W = cube.shape
        output = cube.copy()               # the only large CPU allocation
        p = PAD_WIDTH
        pH, pW = H + 2 * p, W + 2 * p

        device = get_device()
        logger.info(
            "FFT filter: %d bands on %s, padded shape (%d, %d)",
            B, device, pH, pW,
        )

        validity_bool = validity.astype(bool)             # (B, H, W)
        band_means = _band_means(cube, validity_bool)     # (B,)

        # Lives on-device for the whole loop (~18 MB).
        notch_t = torch.from_numpy(notch).to(device=device, dtype=torch.complex64)
        batch_size = _choose_batch_size(B, pH, pW, device)

        diag_rep = diag.representative_band_index if diag is not None else None
        t_prep = 0.0
        t_fft = 0.0
        t_write = 0.0

        for start in range(0, B, batch_size):
            end = min(start + batch_size, B)
            n = end - start
            t0 = time.time()
            padded = _fill_and_pad(
                cube[start:end], validity_bool[start:end],
                band_means[start:end], p, device,
            )
            t_prep += time.time() - t0

            t0 = time.time()
            corrected, filtered = _notch_batch(padded, notch_t)
            t_fft += time.time() - t0

            # Crop back to the original frame and restore only the valid
            # pixels; invalid ones keep their original values from the copy.
            t0 = time.time()
            cropped_np = corrected[:, p:p + H, p:p + W].cpu().numpy()
            val_np = validity_bool[start:end]
            for i in range(n):
                output[start + i][val_np[i]] = cropped_np[i][val_np[i]]
            t_write += time.time() - t0

            if diag_rep is not None and start <= diag_rep < end:
                diag.log_power_after = np.log(
                    np.abs(filtered[diag_rep - start].cpu().numpy()) ** 2 + 1e-10
                )

            del padded, filtered, corrected, cropped_np

            if start == 0:
                logger.info("FFT filter: first batch done (%d bands)", n)

        # Diagnostics: before/after for rep band
        if diag is not None and diag_rep is not None:
            diag.band_before = cube[diag_rep].copy()
            diag.band_after = output[diag_rep].copy()

        logger.info(
            "FFT filter: done — %d bands on %s | prep=%.2fs fft=%.2fs write=%.2fs",
            B, device, t_prep, t_fft, t_write,
        )
        return output
