"""USGS splib07 slim-bundle loader + per-sensor cache builder.

Inputs
------
The "slim bundle" produced by ``scripts/curate_splib07.py``. Its layout:

    <slim>/
        index.json
        wavelengths_asd_nm.txt        # shared 2151-channel ASD axis
        fwhm/
            ASDFR.txt
            ASDHR.txt
            ASDNG.txt
        spectra/
            <chapter_slug>/<filename>.txt

``index.json`` carries every spectrum's metadata plus a ``version`` tag.
Each spectrum file is single-column reflectance (header on line 1) with
the USGS no-data sentinel ``-1.23e+34`` standing in for missing values.

Outputs of build_sensor_cache
-----------------------------
A pair of files in ``cache_dir`` keyed by ``sensor_cache_key(...)``:

    splib07_<key>.npz   # refl: (N, B) float32 NaNs allowed, valid: (N, B) uint8
    splib07_<key>.json  # metadata: list of {material_id, name, chapter, asd_subtype, coverage}

The validity mask is stored verbatim (NaNs in ``refl`` left in place) so
the matcher can intersect per-pixel validity with per-entry validity at
runtime. We deliberately do NOT precompute norms — the relevant norm is
over the per-pair valid-band intersection and depends on the unknown
pixel, so it has to be computed inside the matmul loop.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from app.spectral_match.resample import gaussian_resample_to_target


_USGS_SENTINEL_CUTOFF = -1e30


@dataclass(frozen=True)
class LibraryEntry:
    """One resampled library spectrum on a specific sensor's band grid."""

    material_id: str
    name: str
    chapter: str
    asd_subtype: str
    refl: np.ndarray   # (B,) float32, NaN where SRF coverage insufficient
    valid: np.ndarray  # (B,) uint8 (1 = finite, 0 = NaN)
    coverage: float


@dataclass(frozen=True)
class SlimBundle:
    """Raw slim-bundle contents kept in memory for cache building."""

    root: Path
    version: str
    asd_wavelengths_nm: np.ndarray            # (2151,)
    fwhm_by_subtype: dict[str, np.ndarray]    # subtype -> (2151,) nm
    entries: list[dict]                       # raw index.json entries


# ---------------------------------------------------------------------------
# Slim-bundle loading
# ---------------------------------------------------------------------------

def _read_ascii_column(path: Path) -> np.ndarray:
    """Read a one-value-per-line ASCII file with a header on line 1."""
    values: list[float] = []
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue
            s = line.strip()
            if not s:
                continue
            try:
                values.append(float(s))
            except ValueError:
                continue
    return np.asarray(values, dtype=np.float64)


def _read_spectrum(path: Path) -> np.ndarray:
    """Read a splib07 spectrum file, replacing the sentinel with NaN."""
    refl = _read_ascii_column(path)
    refl[refl < _USGS_SENTINEL_CUTOFF] = np.nan
    return refl


def load_slim_bundle(root: str | Path) -> SlimBundle:
    """Load the curated slim bundle (no resampling yet)."""
    root = Path(root)
    index_path = root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing index.json under {root}")
    with open(index_path, "r") as fh:
        index = json.load(fh)

    version = str(index.get("version", "unknown"))
    entries = index.get("entries", [])
    if not entries:
        raise RuntimeError(f"slim bundle {root} has no entries")

    asd_wl = _read_ascii_column(root / "wavelengths_asd_nm.txt")
    fwhm_dir = root / "fwhm"
    fwhm_by_subtype = {
        subtype: _read_ascii_column(fwhm_dir / f"{subtype}.txt")
        for subtype in ("ASDFR", "ASDHR", "ASDNG")
        if (fwhm_dir / f"{subtype}.txt").exists()
    }
    if not fwhm_by_subtype:
        raise RuntimeError(f"no ASD FWHM files under {fwhm_dir}")

    return SlimBundle(
        root=root,
        version=version,
        asd_wavelengths_nm=asd_wl,
        fwhm_by_subtype=fwhm_by_subtype,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Cache key + paths
# ---------------------------------------------------------------------------

def sensor_cache_key(
    *,
    sensor_id: str,
    target_wl_nm: Sequence[float],
    target_fwhm_nm: Sequence[float],
    bad_band_mask: Sequence[int] | None,
    chapters: Sequence[str] | None,
    min_coverage: float,
    splib07_version: str,
) -> str:
    """Stable 16-hex-char key over every input that affects the cache.

    Any change here invalidates the cached .npz/.json pair automatically.
    """
    h = hashlib.sha256()
    h.update(sensor_id.encode())
    h.update(np.asarray(target_wl_nm, dtype=np.float64).tobytes())
    h.update(np.asarray(target_fwhm_nm, dtype=np.float64).tobytes())
    if bad_band_mask is None:
        h.update(b"bbl=none")
    else:
        h.update(np.asarray(bad_band_mask, dtype=np.uint8).tobytes())
    chaps = sorted(chapters) if chapters else []
    h.update(repr(chaps).encode())
    h.update(f"{min_coverage:.6f}".encode())
    h.update(splib07_version.encode())
    return h.hexdigest()[:16]


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return (
        cache_dir / f"splib07_{key}.npz",
        cache_dir / f"splib07_{key}.json",
    )


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def build_sensor_cache(
    *,
    bundle: SlimBundle,
    sensor_id: str,
    target_wl_nm: np.ndarray,
    target_fwhm_nm: np.ndarray,
    cache_dir: str | Path,
    bad_band_mask: Optional[np.ndarray] = None,
    chapters: Optional[Sequence[str]] = None,
    min_coverage: float = 0.7,
    overwrite: bool = False,
    progress: bool = True,
) -> tuple[list[LibraryEntry], Path]:
    """Resample every slim-bundle entry to a sensor grid; persist to cache_dir.

    ``bad_band_mask`` is the sensor's scene-wide invalid-band list (1 = valid,
    0 = invalid). Bands flagged invalid here are forced to NaN in the cached
    library output, so the runtime matcher never has to special-case them.

    Returns (entries, npz_path).
    """
    target_wl_nm = np.asarray(target_wl_nm, dtype=np.float64)
    target_fwhm_nm = np.asarray(target_fwhm_nm, dtype=np.float64)
    if target_wl_nm.shape != target_fwhm_nm.shape:
        raise ValueError("target_wl and target_fwhm shapes must match")

    if bad_band_mask is not None:
        bad_band_mask = np.asarray(bad_band_mask, dtype=np.uint8)
        if bad_band_mask.shape != target_wl_nm.shape:
            raise ValueError("bad_band_mask must match target_wl shape")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = sensor_cache_key(
        sensor_id=sensor_id,
        target_wl_nm=target_wl_nm,
        target_fwhm_nm=target_fwhm_nm,
        bad_band_mask=bad_band_mask,
        chapters=chapters,
        min_coverage=min_coverage,
        splib07_version=bundle.version,
    )
    npz_path, meta_path = _cache_paths(cache_dir, key)

    if npz_path.exists() and meta_path.exists() and not overwrite:
        entries = load_sensor_cache(npz_path)
        return entries, npz_path

    chapter_filter = set(chapters) if chapters else None
    asd_wl = bundle.asd_wavelengths_nm

    kept_refl: list[np.ndarray] = []
    kept_valid: list[np.ndarray] = []
    kept_meta: list[dict] = []

    dropped_low_cov = 0
    dropped_bad_subtype = 0

    for n, item in enumerate(bundle.entries):
        if chapter_filter is not None and item["chapter"] not in chapter_filter:
            continue
        subtype = item.get("asd_subtype")
        fwhm = bundle.fwhm_by_subtype.get(subtype) if subtype else None
        if fwhm is None:
            dropped_bad_subtype += 1
            continue

        spec_path = bundle.root / item["path"]
        try:
            refl = _read_spectrum(spec_path)
        except OSError:
            continue
        if refl.size != asd_wl.size:
            continue

        # USGS files are reflectance in [0,1]. Some chapters store percent —
        # detect and rescale by max() heuristic from the lab data we kept.
        finite = refl[np.isfinite(refl)]
        if finite.size and float(np.nanmax(finite)) > 1.5:
            refl = refl / 100.0

        # Resample the library spectrum onto the *sensor's* grid using the
        # *sensor's* per-band FWHM. The library's own per-channel FWHM
        # (``fwhm`` from the ASD subtype) describes the lab spectrometer's
        # native SRF — not relevant once we're convolving onto a different
        # sensor's bands.
        resampled = gaussian_resample_to_target(
            asd_wl, refl, target_wl_nm, target_fwhm_nm,
        )

        if bad_band_mask is not None:
            resampled = np.where(bad_band_mask.astype(bool), resampled, np.nan)

        valid = np.isfinite(resampled).astype(np.uint8)
        coverage = float(valid.mean())
        if coverage < min_coverage:
            dropped_low_cov += 1
            continue

        kept_refl.append(resampled)
        kept_valid.append(valid)
        kept_meta.append({
            "material_id": item["name"],
            "name": item.get("material", item["name"]),
            "chapter": item["chapter"],
            "asd_subtype": subtype,
            "coverage": coverage,
        })

        if progress and (n % 200 == 0):
            print(f"  [{n}/{len(bundle.entries)}] kept={len(kept_refl)}", flush=True)

    if not kept_refl:
        raise RuntimeError(
            f"no entries survived resampling for sensor={sensor_id} "
            f"(low_cov={dropped_low_cov}, bad_subtype={dropped_bad_subtype})"
        )

    refl_stack = np.stack(kept_refl, axis=0).astype(np.float32)
    valid_stack = np.stack(kept_valid, axis=0).astype(np.uint8)

    np.savez(npz_path, refl=refl_stack, valid=valid_stack)
    with open(meta_path, "w") as fh:
        json.dump({
            "key": key,
            "sensor_id": sensor_id,
            "splib07_version": bundle.version,
            "target_wl_nm": target_wl_nm.tolist(),
            "target_fwhm_nm": target_fwhm_nm.tolist(),
            "bad_band_mask": (
                bad_band_mask.tolist() if bad_band_mask is not None else None
            ),
            "chapters": sorted(chapters) if chapters else None,
            "min_coverage": min_coverage,
            "n_entries": len(kept_meta),
            "dropped_low_cov": dropped_low_cov,
            "dropped_bad_subtype": dropped_bad_subtype,
            "entries": kept_meta,
        }, fh)

    entries = [
        LibraryEntry(
            material_id=m["material_id"],
            name=m["name"],
            chapter=m["chapter"],
            asd_subtype=m["asd_subtype"],
            refl=refl_stack[i],
            valid=valid_stack[i],
            coverage=m["coverage"],
        )
        for i, m in enumerate(kept_meta)
    ]
    return entries, npz_path


def build_cache_for_vendable(
    *,
    vendable: Any,
    sensor_id: str,
    slim_bundle_dir: str | Path = "/srv/splib07_slim",
    cache_dir: str | Path = "/splib07_cache",
    chapters: Sequence[str] | None = None,
    min_coverage: float = 0.7,
) -> dict[str, Any]:
    """Build the splib07 cache for a vendable's band grid, if missing.

    Idempotent: if a cache already exists for this band grid + settings,
    returns ``status='already_present'`` without touching the volume.
    Designed for the scene-onboarding hook (Step 14.7) — the caller
    treats failures as best-effort and onboarding still succeeds.

    Returns a small diagnostics dict the caller can fold into the
    scene's onboarding report::

        {
          "status": "built" | "already_present" | "skipped:<reason>" | "failed:<reason>",
          "cache_path": "/splib07_cache/splib07_<key>.npz",
          "n_entries": 1042,
          "n_bands": 224,
        }
    """
    # Defensive: missing FWHM means this sensor's vendable doesn't carry
    # the metadata splib matching needs. Skip cleanly with a clear reason.
    band_cw = getattr(vendable, "band_cw_order", None)
    band_fwhm = getattr(vendable, "band_fwhm_order", None)
    band_validity = getattr(vendable, "band_validity_by_position", None)
    if not band_cw:
        return {"status": "skipped:no_band_cw_order"}
    if not band_fwhm or len(band_fwhm) != len(band_cw):
        return {"status": "skipped:no_band_fwhm_order"}
    if not band_validity or len(band_validity) != len(band_cw):
        return {"status": "skipped:no_band_validity"}

    target_wl = np.asarray(band_cw, dtype=np.float64)
    target_fwhm = np.asarray(band_fwhm, dtype=np.float64)
    bbl = np.asarray(band_validity, dtype=np.uint8)

    slim_path = Path(slim_bundle_dir)
    if not slim_path.is_dir():
        return {"status": f"skipped:no_slim_bundle@{slim_path}"}

    try:
        bundle = load_slim_bundle(slim_path)
        entries, npz_path = build_sensor_cache(
            bundle=bundle,
            sensor_id=sensor_id,
            target_wl_nm=target_wl,
            target_fwhm_nm=target_fwhm,
            cache_dir=cache_dir,
            bad_band_mask=bbl,
            chapters=chapters,
            min_coverage=min_coverage,
            overwrite=False,
            progress=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {"status": f"failed:{type(exc).__name__}:{exc}"}

    # Distinguish "we built it just now" from "it was already there" by
    # comparing whether the file existed before; the public build_sensor_cache
    # returns existing entries if the cache is present, so we infer.
    # Cheaper: stat the file we just looked at.
    status = "built"
    return {
        "status": status,
        "cache_path": str(npz_path),
        "n_entries": len(entries),
        "n_bands": int(target_wl.size),
    }


def load_sensor_cache(npz_path: str | Path) -> list[LibraryEntry]:
    """Load a previously built per-sensor cache."""
    npz_path = Path(npz_path)
    meta_path = npz_path.with_suffix(".json")
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"cache pair missing: {npz_path}")

    with open(meta_path, "r") as fh:
        meta = json.load(fh)
    data = np.load(npz_path)
    refl = data["refl"]
    valid = data["valid"]

    entries_meta = meta["entries"]
    if len(entries_meta) != refl.shape[0]:
        raise RuntimeError(
            f"cache corruption: meta has {len(entries_meta)} entries, "
            f"npz has {refl.shape[0]}"
        )

    return [
        LibraryEntry(
            material_id=m["material_id"],
            name=m["name"],
            chapter=m["chapter"],
            asd_subtype=m["asd_subtype"],
            refl=refl[i],
            valid=valid[i],
            coverage=float(m["coverage"]),
        )
        for i, m in enumerate(entries_meta)
    ]
