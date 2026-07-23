"""Build a per-sensor splib07 cache (.npz + .json) from the slim bundle.

Resamples every spectrum in the curated slim bundle onto a specific
sensor's wavelength grid using that sensor's per-band FWHM, and writes
a content-addressed cache pair into ``--cache-dir``.

The cache key encodes everything that affects the output:
    (sensor_id, target_wl, target_fwhm, bad_band_mask, chapters,
     min_coverage, splib07_version)

so running this command twice with the same arguments is a no-op (unless
``--overwrite``). Changing any input rebuilds a separate cache file.

Sensor inputs come from a small JSON spec the caller writes — one file
per sensor — containing native wavelengths (nm), per-band FWHM (nm), and
optional bad-band mask. Example layout:

    {
      "sensor_id": "aviris_ng",
      "target_wl_nm":  [380.0, 385.0, ...],
      "target_fwhm_nm": [6.1, 6.1, ...],
      "bad_band_mask": [1, 1, 0, 0, 1, ...]    // optional, 1=valid
    }

These specs are emitted at onboarding time alongside the vendable —
that's a future hook in the data pipeline; for now the operator writes
them by hand from the sensor's reference.

Usage:

    python scripts/build_splib_sensor_cache.py \\
        --slim       data/splib07_slim \\
        --sensor-spec data/sensor_specs/aviris_ng.json \\
        --cache-dir  data/splib07_cache \\
        --chapters minerals artificial soils vegetation organics \\
        --min-coverage 0.7
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from app.spectral_match.library import build_sensor_cache, load_slim_bundle


logger = logging.getLogger("build_splib_sensor_cache")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--slim", required=True, type=Path,
                   help="path to the curated slim bundle directory")
    p.add_argument("--sensor-spec", required=True, type=Path,
                   help="JSON spec with sensor_id, target_wl_nm, target_fwhm_nm, "
                        "and optional bad_band_mask")
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="directory where per-sensor cache files are written")
    p.add_argument("--chapters", nargs="*", default=None,
                   help="restrict to these chapter slugs "
                        "(minerals, artificial, soils, vegetation, organics, liquids, coatings); "
                        "default: all chapters in the bundle")
    p.add_argument("--min-coverage", type=float, default=0.7,
                   help="drop library entries whose resampled-valid fraction is "
                        "below this; default 0.7")
    p.add_argument("--overwrite", action="store_true",
                   help="rebuild even if a matching cache already exists")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.slim.is_dir():
        logger.error("slim bundle dir not found: %s", args.slim)
        return 2
    if not args.sensor_spec.is_file():
        logger.error("sensor spec not found: %s", args.sensor_spec)
        return 2

    with open(args.sensor_spec, "r") as fh:
        spec = json.load(fh)

    sensor_id = spec["sensor_id"]
    target_wl = np.asarray(spec["target_wl_nm"], dtype=np.float64)
    target_fwhm = np.asarray(spec["target_fwhm_nm"], dtype=np.float64)
    bad_band_mask = spec.get("bad_band_mask")
    bad_band_mask_arr = (
        np.asarray(bad_band_mask, dtype=np.uint8)
        if bad_band_mask is not None else None
    )

    logger.info("loading slim bundle from %s", args.slim)
    bundle = load_slim_bundle(args.slim)
    logger.info(
        "slim bundle v=%s entries=%d ASD-wl=%d FWHM-subtypes=%s",
        bundle.version, len(bundle.entries),
        bundle.asd_wavelengths_nm.size,
        sorted(bundle.fwhm_by_subtype.keys()),
    )

    logger.info(
        "building cache for sensor=%s bands=%d min_coverage=%.2f chapters=%s",
        sensor_id, target_wl.size, args.min_coverage,
        args.chapters or "(all)",
    )

    entries, npz_path = build_sensor_cache(
        bundle=bundle,
        sensor_id=sensor_id,
        target_wl_nm=target_wl,
        target_fwhm_nm=target_fwhm,
        cache_dir=args.cache_dir,
        bad_band_mask=bad_band_mask_arr,
        chapters=args.chapters,
        min_coverage=args.min_coverage,
        overwrite=args.overwrite,
        progress=args.verbose,
    )

    by_chapter: dict[str, int] = {}
    for e in entries:
        by_chapter[e.chapter] = by_chapter.get(e.chapter, 0) + 1
    logger.info("wrote %s with %d entries", npz_path, len(entries))
    for chap, n in sorted(by_chapter.items()):
        logger.info("  %-12s %5d", chap, n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
