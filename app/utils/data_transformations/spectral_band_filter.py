"""
Wavelength-based band exclusion utility.

Combines band validity flags with wavelength exclusion ranges to produce
a clean list of usable band indices. Sensor-agnostic — pass different
exclusion ranges for different instruments.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Standard PRISMA exclusion ranges (nm)
DEFAULT_EXCLUSION_RANGES: List[Tuple[float, float]] = [
    (0, 450),          # low SNR, detector noise
    (912, 978),        # water vapor
    (1131, 1152),      # water vapor
    (1328, 1492),      # water vapor, deep
    (1784, 1967),      # water vapor + CO2
]


class SpectralBandFilter:
    """
    Filters bands by validity flags and wavelength exclusion ranges.
    """

    def __init__(
        self,
        band_wavelengths: List[float],
        band_validity_flags: List[int],
        exclusion_ranges: List[Tuple[float, float]] = None,
    ):
        self._wavelengths = band_wavelengths
        self._flags = band_validity_flags
        self._exclusion_ranges = (
            exclusion_ranges if exclusion_ranges is not None
            else DEFAULT_EXCLUSION_RANGES
        )
        self._summary_logged = False
        self._good_indices: List[int] | None = None

    def get_good_band_indices(self) -> List[int]:
        """
        Returns just the good band indices.
        """
        if self._good_indices is not None:
            return self._good_indices

        good = []
        for i in range(len(self._wavelengths)):
            if self._flags[i] != 1:
                continue
            wl = self._wavelengths[i]
            if any(lo <= wl <= hi for lo, hi in self._exclusion_ranges):
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

        dropped_by_wl = []
        drop_reasons = {}
        for i in range(total):
            if self._flags[i] != 1:
                continue
            wl = self._wavelengths[i]
            for lo, hi in self._exclusion_ranges:
                if lo <= wl <= hi:
                    dropped_by_wl.append(i)
                    drop_reasons[i] = (lo, hi)
                    break

        good = self.get_good_band_indices()

        return {
            "total_bands": total,
            "dropped_by_flags": {"count": len(dropped_by_flag), "indices": dropped_by_flag},
            "dropped_by_wavelength": {
                "count": len(dropped_by_wl),
                "indices": dropped_by_wl,
                "ranges": drop_reasons,
            },
            "surviving": len(good),
        }

    def _log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "SpectralBandFilter: %d total → %d dropped by flags, "
            "%d dropped by wavelength → %d surviving",
            s["total_bands"],
            s["dropped_by_flags"]["count"],
            s["dropped_by_wavelength"]["count"],
            s["surviving"],
        )
