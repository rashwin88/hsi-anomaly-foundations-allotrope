"""
One place to resolve per-band normalisation statistics.

Every model that has a PixelNormalize layer needs (mean, std) at build time, so
this exact block used to be copy-pasted into four inferencers and six trainers -
ten near-identical loaders that had already drifted in their logging and could
drift in their behaviour.

Two sources, in priority order:

  1. An in-memory PixelStatsOverride, used for uncalibrated sensors. HotSat-1
     ships raw DN (~5000 +/- 400) against checkpoints normalised for Celsius
     (~290 +/- 10), so the stats are recomputed from the scene itself. Inference
     only - trainers never pass one.
  2. A JSON file at pixel_stats_path, holding {"mean": [...], "std": [...]}.
     These are the baked dataset statistics under app/constants/.

Returns (None, None) when neither is set, which the model constructors read as
"skip normalisation entirely".

The override deliberately wins outright rather than merging: when the caller
sets one it also passes pixel_stats_path=None, and silently falling back to
disk would hand the model the wrong distribution for the sensor.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("allotrope.pixel_stats")

# Above this many bands, log counts rather than the values themselves - the
# hyperspectral grid is 165 entries and would bury the rest of the log.
_MAX_BANDS_TO_LOG = 4


def resolve_pixel_stats(
    stats_path: str | None,
    override: Any | None = None,
) -> tuple[list[float] | None, list[float] | None]:
    """
    Resolve (pixel_mean, pixel_std) from an override or a stats JSON file.

    `override` is typed loosely on purpose: it only needs `.mean`, `.std` and
    `.source`, and typing it as PixelStatsOverride would make this module
    import from app.models.training, which the components layer does not
    otherwise depend on.
    """
    if override is not None:
        mean, std = list(override.mean), list(override.std)
        logger.info(
            "Pixel stats overridden per-scene (source=%s, %d bands)",
            getattr(override, "source", "unknown"),
            len(mean),
        )
        return mean, std

    if stats_path is None:
        return None, None

    with open(stats_path) as handle:
        stats = json.load(handle)
    mean, std = stats["mean"], stats["std"]

    if len(mean) <= _MAX_BANDS_TO_LOG:
        logger.info("Pixel stats loaded: mean=%s, std=%s", mean, std)
    else:
        logger.info("Pixel stats loaded: %d means, %d stds", len(mean), len(std))
    return mean, std


def resolve_wavelength_stats(stats_path: str) -> tuple[list[float], list[float], list[float]]:
    """(wavelengths_nm, mean, std) from a stats file that carries wavelengths - the
    `stats.json` scripts/shard_spectralearth.py writes beside the shards. Raises ValueError
    if the file has no `wavelengths_nm` or the three lists differ in length: a band-position
    stats file (the 165-entry app/constants ones) cannot be read off at other wavelengths."""
    with open(stats_path) as handle:
        stats = json.load(handle)
    if "wavelengths_nm" not in stats:
        raise ValueError(f"{stats_path}: no wavelengths_nm - not a wavelength stats file")
    wl, mean, std = stats["wavelengths_nm"], stats["mean"], stats["std"]
    if not len(wl) == len(mean) == len(std):
        raise ValueError(f"{stats_path}: {len(wl)} wavelengths, {len(mean)} means, {len(std)} stds")
    logger.info("Wavelength stats loaded: %d bands, %.0f-%.0f nm", len(wl), min(wl), max(wl))
    return wl, mean, std
