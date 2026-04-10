"""
Wavelength-based band exclusion utility.

Combines band validity flags with wavelength exclusion ranges to produce
a clean list of usable band indices. Sensor-agnostic — pass different
exclusion ranges for different instruments.

Filtering stages (applied in order):
1. Sensor-flagged bad bands (validity flag != 1)
2. Wavelength exclusion ranges (atmospheric absorption windows)
3. Noisy edge bands (first/last N bands of each detector)
4. Coverage-aware pruning (bands with < threshold% valid pixels)
"""

import logging
from typing import List, Tuple, Optional

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

logger = logging.getLogger(__name__)

# Standard PRISMA exclusion ranges (nm)
DEFAULT_EXCLUSION_RANGES: List[Tuple[float, float]] = [
    (0, 450),          # low SNR, detector noise
    (912, 978),        # water vapor
    (1131, 1152),      # water vapor
    (1328, 1492),      # water vapor, deep
    (1784, 1967),      # water vapor + CO2
]

# Atmospheric absorption windows common to all sensors (nm)
ATMOSPHERIC_EXCLUSION_RANGES: List[Tuple[float, float]] = [
    (1350, 1450),      # water vapor absorption
    (1800, 1950),      # water vapor + CO2 absorption
]


class SpectralBandFilter:
    """
    Filters bands by validity flags, wavelength exclusion ranges,
    detector edge trimming, and coverage-aware pruning.
    """

    def __init__(
        self,
        band_wavelengths: List[float],
        band_validity_flags: List[int],
        exclusion_ranges: List[Tuple[float, float]] = None,
        spectral_families: Optional[List[SpectralFamily]] = None,
        edge_bands_to_trim: int = 0,
        band_level_validity_scores: Optional[List[float]] = None,
        min_valid_pixel_pct: float = 20.0,
    ):
        """
        Args:
            band_wavelengths: Center wavelength of each band (nm).
            band_validity_flags: 1 = valid, 0 = flagged bad by sensor.
            exclusion_ranges: Wavelength ranges to exclude. If None, uses
                DEFAULT_EXCLUSION_RANGES.
            spectral_families: Per-band spectral family (VNIR/SWIR). Required
                when edge_bands_to_trim > 0 to identify detector boundaries.
            edge_bands_to_trim: Number of bands to trim from each end of each
                detector (VNIR, SWIR). 0 = no trimming.
            band_level_validity_scores: Per-band valid pixel percentage
                (0.0–100.0). Required for coverage-aware pruning.
            min_valid_pixel_pct: Minimum valid pixel percentage to keep a band.
                Bands below this threshold are dropped. Default 20.0%.
        """
        self._wavelengths = band_wavelengths
        self._flags = band_validity_flags
        self._exclusion_ranges = (
            exclusion_ranges if exclusion_ranges is not None
            else DEFAULT_EXCLUSION_RANGES
        )
        self._spectral_families = spectral_families
        self._edge_bands_to_trim = edge_bands_to_trim
        self._validity_scores = band_level_validity_scores
        self._min_valid_pixel_pct = min_valid_pixel_pct
        self._summary_logged = False
        self._good_indices: List[int] | None = None

    def _compute_edge_band_indices(self) -> set:
        """
        Returns the set of indices that fall within the first/last N bands
        of each detector (spectral family).
        """
        if self._edge_bands_to_trim <= 0 or self._spectral_families is None:
            return set()

        edge_indices = set()
        # Group contiguous runs of each family
        families_seen = {}
        for i, fam in enumerate(self._spectral_families):
            families_seen.setdefault(fam, []).append(i)

        n = self._edge_bands_to_trim
        for fam, indices in families_seen.items():
            sorted_idx = sorted(indices)
            # Trim first N and last N of each detector
            edge_indices.update(sorted_idx[:n])
            edge_indices.update(sorted_idx[-n:])

        return edge_indices

    def get_good_band_indices(self) -> List[int]:
        """
        Returns the indices of bands that survive all four filtering stages.
        """
        if self._good_indices is not None:
            return self._good_indices

        edge_set = self._compute_edge_band_indices()

        good = []
        for i in range(len(self._wavelengths)):
            # Stage 1: sensor-flagged bad bands
            if self._flags[i] != 1:
                continue
            # Stage 2: wavelength exclusion ranges
            wl = self._wavelengths[i]
            if any(lo <= wl <= hi for lo, hi in self._exclusion_ranges):
                continue
            # Stage 3: edge band trimming
            if i in edge_set:
                continue
            # Stage 4: coverage-aware pruning
            if self._validity_scores is not None:
                if self._validity_scores[i] < self._min_valid_pixel_pct:
                    continue
            good.append(i)

        self._good_indices = good

        if not self._summary_logged:
            self._log_summary()
            self._summary_logged = True

        return good

    def get_good_band_wavelengths(self) -> List[float]:
        return [self._wavelengths[i] for i in self.get_good_band_indices()]

    def summary(self) -> dict:
        total = len(self._wavelengths)

        dropped_by_flag = [i for i, v in enumerate(self._flags) if v != 1]

        # For wavelength drops, only consider bands that passed flag filter
        flag_survivors = {i for i in range(total) if self._flags[i] == 1}

        dropped_by_wl = []
        drop_reasons = {}
        for i in flag_survivors:
            wl = self._wavelengths[i]
            for lo, hi in self._exclusion_ranges:
                if lo <= wl <= hi:
                    dropped_by_wl.append(i)
                    drop_reasons[i] = (lo, hi)
                    break

        # For edge drops, only consider bands surviving flags + wavelength
        wl_survivors = flag_survivors - set(dropped_by_wl)
        edge_set = self._compute_edge_band_indices()
        dropped_by_edge = sorted(wl_survivors & edge_set)

        # For coverage drops, only consider bands surviving all prior stages
        edge_survivors = wl_survivors - edge_set
        dropped_by_coverage = []
        if self._validity_scores is not None:
            dropped_by_coverage = sorted(
                i for i in edge_survivors
                if self._validity_scores[i] < self._min_valid_pixel_pct
            )

        good = self.get_good_band_indices()

        return {
            "total_bands": total,
            "dropped_by_flags": {"count": len(dropped_by_flag), "indices": dropped_by_flag},
            "dropped_by_wavelength": {
                "count": len(dropped_by_wl),
                "indices": sorted(dropped_by_wl),
                "ranges": drop_reasons,
            },
            "dropped_by_edge": {"count": len(dropped_by_edge), "indices": dropped_by_edge},
            "dropped_by_coverage": {
                "count": len(dropped_by_coverage),
                "indices": dropped_by_coverage,
                "threshold_pct": self._min_valid_pixel_pct,
            },
            "surviving": len(good),
        }

    def _log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "SpectralBandFilter: %d total → %d dropped by flags, "
            "%d dropped by wavelength, %d dropped by edge trim, "
            "%d dropped by coverage (<%g%%) → %d surviving",
            s["total_bands"],
            s["dropped_by_flags"]["count"],
            s["dropped_by_wavelength"]["count"],
            s["dropped_by_edge"]["count"],
            s["dropped_by_coverage"]["count"],
            self._min_valid_pixel_pct,
            s["surviving"],
        )
