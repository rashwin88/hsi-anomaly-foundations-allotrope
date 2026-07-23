"""
USGS splib07 spectral library loader.

This is the modularized, full-library replacement for the man-made-only
keyword filter in the original script. Behaviour:

    1. Walks the splib07 directory tree (one of the cv* convolved variants,
       e.g. ASCIIdata_splib07b_cvASD).
    2. Identifies wavelength companion files per directory.
    3. For each spectrum file, derives:
        - material name (from the file's header line)
        - material_id  (stable across runs; the file's basename without .txt)
        - category     (from the nearest 'Chapter*' ancestor folder, e.g.
                        ChapterM_Minerals -> 'Minerals'; if no Chapter*
                        ancestor, category = 'Uncategorized')
    4. Resamples each spectrum to the target sensor wavelengths using
       Gaussian SRFs (from spectral.resample).
    5. Returns a list of LibraryEntry plus the stacked (N, n_bands) array.

splib07 directory layout (typical)
----------------------------------
ASCIIdata_splib07b_cvASD/
    ChapterA_ArtificialMaterials/
        s07_ASD_Aluminum_metal_*.txt          <-- spectrum files
        s07_ASD_Wavelengths_*_2151_ch_ASDa.txt <-- wavelength file (ONE per folder)
    ChapterC_Coatings/
        ...
    ChapterL_Liquids/
    ChapterM_Minerals/
    ChapterO_OrganicCompounds/
    ChapterS_SoilsAndMixtures/
    ChapterV_Vegetation/

Each spectrum file is single-column reflectance; wavelengths come from the
companion file in the same (or parent) directory, in microns. We convert
to nm.

USGS no-data sentinel: -1.23e+34, replaced with NaN.
Some entries are stored on a 0-100 scale (percent reflectance); we detect
and convert to 0-1 before resampling.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .resample import gaussian_resample_to_target


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    """One entry in the resampled library, ready for SAM."""
    material_id: str        # stable identifier (basename without .txt)
    name: str               # human-readable name from header line
    category: str           # 'Minerals', 'Vegetation', 'ArtificialMaterials', ...
    source_path: str        # original spectrum file (for traceability)
    refl: np.ndarray        # (n_bands_target,) float32 resampled reflectance
    coverage: float         # fraction of target bands populated by the SRF


# ---------------------------------------------------------------------------
# File parsing
# ---------------------------------------------------------------------------

# splib07 chapter folder name pattern: Chapter<X>_<Name>
_CHAPTER_RE = re.compile(r"^Chapter[A-Z]+_(.+)$")


def _category_from_path(spec_path: str) -> str:
    """Walk up from the spectrum file looking for a Chapter* folder."""
    p = Path(spec_path).resolve()
    for ancestor in p.parents:
        m = _CHAPTER_RE.match(ancestor.name)
        if m:
            return m.group(1)
    return "Uncategorized"


def _read_wavelength_file(filepath: str) -> Optional[np.ndarray]:
    """
    Read a splib07 wavelength companion file. Returns wavelengths in nm.
    Header line is skipped; remaining lines are floats in microns.
    """
    values: List[float] = []
    try:
        with open(filepath, "r", errors="replace") as fh:
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
    except OSError:
        return None
    if not values:
        return None
    wl = np.array(values, dtype=np.float64)
    # splib07 wavelength files are in microns by convention. If the max is
    # well below 100 we're definitely in microns; convert to nm.
    if np.nanmax(wl) < 100.0:
        wl *= 1000.0
    return wl


def _read_spectrum(filepath: str) -> Tuple[str, Optional[np.ndarray]]:
    """
    Read a splib07 single-column spectrum file.

    Header (line 1) format:
        s07_AV14 Record=5448: Aluminum brushed 293K  ASDFRa AREF
                              ^^^^^^^^^^^^^^^^^^^^^  trailing tokens =
                              the material name      instrument + data type

    Returns (name, refl) where refl has -1.23e+34 sentinels replaced by NaN.
    """
    name = Path(filepath).stem
    refl_list: List[float] = []
    try:
        with open(filepath, "r", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if i == 0:
                    if ":" in line:
                        after_colon = line.split(":", 1)[1].strip()
                        toks = after_colon.split()
                        # The last 2 tokens are usually the instrument code +
                        # data-type code (e.g. "ASDFRa AREF"). Strip them.
                        if len(toks) >= 3:
                            name = " ".join(toks[:-2]).strip() or name
                        elif toks:
                            name = after_colon.strip()
                    continue
                if not line:
                    continue
                try:
                    refl_list.append(float(line))
                except ValueError:
                    continue
    except OSError:
        return name, None

    if not refl_list:
        return name, None

    refl = np.array(refl_list, dtype=np.float64)
    refl[refl < -1e30] = np.nan  # USGS no-data sentinel
    return name, refl


# ---------------------------------------------------------------------------
# Library walk
# ---------------------------------------------------------------------------

def _index_wavelength_files(root: Path) -> dict:
    """Map directory -> wavelength file path by walking the tree."""
    wl_by_dir: dict = {}
    for fp in glob.iglob(str(root / "**" / "*.txt"), recursive=True):
        if "wavelength" in os.path.basename(fp).lower():
            wl_by_dir[str(Path(fp).parent)] = fp
    return wl_by_dir


def _resolve_wl_file(spec_dir: str, wl_by_dir: dict) -> Optional[str]:
    """
    Find the wavelength file for a given spectrum directory.

    Most splib07 layouts put the wavelength file alongside the spectra in
    each chapter folder. Some put it one level up. We check current then
    up to two parents.
    """
    candidates = [spec_dir, str(Path(spec_dir).parent), str(Path(spec_dir).parent.parent)]
    for d in candidates:
        if d in wl_by_dir:
            return wl_by_dir[d]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _cache_key(
    splib07_dir: str,
    target_wl: np.ndarray,
    target_fwhm: np.ndarray,
    restrict_to_categories: Optional[List[str]],
    min_coverage: float,
) -> str:
    """
    Stable hash combining everything that affects the resampled library.
    If any of these change, the cache is invalidated automatically.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(str(Path(splib07_dir).resolve()).encode())
    h.update(target_wl.astype(np.float64).tobytes())
    h.update(target_fwhm.astype(np.float64).tobytes())
    cats = sorted(restrict_to_categories) if restrict_to_categories else []
    h.update(repr(cats).encode())
    h.update(f"{min_coverage:.6f}".encode())
    return h.hexdigest()[:16]


def _try_load_cache(cache_path: Path) -> Optional[List[LibraryEntry]]:
    """Restore a list of LibraryEntry from an .npz + .json sidecar pair."""
    npz_path = cache_path.with_suffix(".npz")
    meta_path = cache_path.with_suffix(".json")
    if not (npz_path.exists() and meta_path.exists()):
        return None
    try:
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        npz = np.load(npz_path)
        refl_stack = npz["refl"]   # (N, n_bands)
        if len(meta) != refl_stack.shape[0]:
            return None
        entries = [
            LibraryEntry(
                material_id=m["material_id"],
                name=m["name"],
                category=m["category"],
                source_path=m["source_path"],
                refl=refl_stack[i].astype(np.float32),
                coverage=float(m["coverage"]),
            )
            for i, m in enumerate(meta)
        ]
        return entries
    except (OSError, KeyError, ValueError):
        return None


def _save_cache(cache_path: Path, entries: List[LibraryEntry]) -> None:
    """Write the library to disk for fast reuse on the next run."""
    import json
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    refl_stack = np.stack([e.refl for e in entries], axis=0).astype(np.float32)
    np.savez(cache_path.with_suffix(".npz"), refl=refl_stack)
    meta = [
        {
            "material_id": e.material_id,
            "name": e.name,
            "category": e.category,
            "source_path": e.source_path,
            "coverage": e.coverage,
        }
        for e in entries
    ]
    with open(cache_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=0)


def load_splib07_library(
    splib07_dir: str,
    target_wl: np.ndarray,
    target_fwhm: np.ndarray,
    restrict_to_categories: Optional[List[str]] = None,
    min_coverage: float = 0.7,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
) -> List[LibraryEntry]:
    """
    Load every spectrum under `splib07_dir`, resample to target sensor bands.

    Parameters
    ----------
    splib07_dir : str
        Path to a cv* convolved splib07 directory (e.g.
        ASCIIdata_splib07b_cvASD), OR a specific chapter subdirectory.
    target_wl : (n_bands,) ndarray
        Target sensor wavelengths in nm.
    target_fwhm : (n_bands,) ndarray
        Target sensor per-band FWHMs in nm.
    restrict_to_categories : list of str, optional
        If provided, only entries whose derived category is in this list
        are returned. Default: load everything.
    min_coverage : float
        Drop entries where < min_coverage of target bands have valid SRF
        output. Defaults to 0.7 (the original script's threshold).
    verbose : bool
        Print progress / counts.
    cache_dir : str, optional
        If set, the resampled library is cached here keyed by
        (splib07_dir, target_wl, target_fwhm, restrict_to_categories,
        min_coverage). Subsequent runs with the same parameters skip the
        per-file walk entirely. None = no caching.

    Returns
    -------
    list[LibraryEntry]
        Sorted by category, then material_id (stable across runs).
    """
    # ----- cache hit fast path ---------------------------------------------
    cache_path = None
    if cache_dir:
        key = _cache_key(splib07_dir, target_wl, target_fwhm,
                         restrict_to_categories, min_coverage)
        cache_path = Path(cache_dir) / f"splib07_resampled_{key}"
        cached = _try_load_cache(cache_path)
        if cached is not None:
            if verbose:
                print(f"\n[lib] cache hit: {cache_path}.npz "
                      f"({len(cached)} entries)")
            return cached
        if verbose:
            print(f"\n[lib] cache miss: will build and save to "
                  f"{cache_path}.npz")

    root = Path(splib07_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"USGS splib07 dir not found: {splib07_dir}")

    if verbose:
        print(f"\n[lib] Walking {root}")

    wl_by_dir = _index_wavelength_files(root)
    if verbose:
        print(f"      indexed {len(wl_by_dir)} wavelength files")

    # Cache: directory -> resolved wavelength array (nm)
    wl_cache: dict = {}

    spec_files: List[str] = []
    for fp in glob.iglob(str(root / "**" / "*.txt"), recursive=True):
        fname = os.path.basename(fp).lower()
        if "wavelength" in fname or "resolution" in fname:
            continue
        if fname.startswith("s07_"):
            spec_files.append(fp)

    if not spec_files:
        raise RuntimeError(
            f"No s07_*.txt spectrum files under {splib07_dir}. "
            "Make sure you point at an extracted ASCIIdata_splib07b_cv* directory."
        )

    if verbose:
        print(f"      found {len(spec_files)} spectrum files")

    cat_filter = set(restrict_to_categories) if restrict_to_categories else None

    entries: List[LibraryEntry] = []
    counters = {
        "no_wl": 0, "len_mismatch": 0, "low_cov": 0,
        "pct_converted": 0, "filtered_out": 0, "kept": 0,
    }

    for fp in sorted(spec_files):
        category = _category_from_path(fp)
        if cat_filter is not None and category not in cat_filter:
            counters["filtered_out"] += 1
            continue

        spec_dir = str(Path(fp).parent)
        if spec_dir not in wl_cache:
            wl_path = _resolve_wl_file(spec_dir, wl_by_dir)
            wl_cache[spec_dir] = (
                _read_wavelength_file(wl_path) if wl_path else None
            )
        lib_wl = wl_cache[spec_dir]
        if lib_wl is None:
            counters["no_wl"] += 1
            continue

        name, refl = _read_spectrum(fp)
        if refl is None:
            continue

        if len(refl) != len(lib_wl):
            counters["len_mismatch"] += 1
            continue

        # Detect 0-100 percent scale and convert to 0-1.
        finite = refl[np.isfinite(refl)]
        if finite.size and np.nanmax(finite) > 1.5:
            refl = refl / 100.0
            counters["pct_converted"] += 1

        # Library wavelengths must be ascending for the SRF function.
        if not np.all(np.diff(lib_wl) > 0):
            order = np.argsort(lib_wl)
            lib_wl_sorted = lib_wl[order]
            refl_sorted = refl[order]
        else:
            lib_wl_sorted = lib_wl
            refl_sorted = refl

        resampled = gaussian_resample_to_target(
            lib_wl_sorted, refl_sorted, target_wl, target_fwhm
        )
        coverage = float(np.isfinite(resampled).mean())
        if coverage < min_coverage:
            counters["low_cov"] += 1
            continue

        # Linear-interpolate the few NaNs (small gaps are typical).
        nan_idx = ~np.isfinite(resampled)
        if nan_idx.any():
            ok = ~nan_idx
            resampled[nan_idx] = np.interp(
                target_wl[nan_idx], target_wl[ok], resampled[ok]
            )
        resampled = np.clip(resampled, 0.0, 1.0)

        entries.append(LibraryEntry(
            material_id=Path(fp).stem,
            name=name,
            category=category,
            source_path=fp,
            refl=resampled.astype(np.float32),
            coverage=coverage,
        ))
        counters["kept"] += 1

    entries.sort(key=lambda e: (e.category, e.material_id))

    if verbose:
        print(f"      kept            : {counters['kept']}")
        print(f"      filtered out    : {counters['filtered_out']}")
        print(f"      no wavelength   : {counters['no_wl']}")
        print(f"      length mismatch : {counters['len_mismatch']}")
        print(f"      low coverage    : {counters['low_cov']}")
        print(f"      pct->01 rescaled: {counters['pct_converted']}")
        # Per-category breakdown helps spot if a chapter wasn't found.
        cats: dict = {}
        for e in entries:
            cats[e.category] = cats.get(e.category, 0) + 1
        for cat, n in sorted(cats.items()):
            print(f"        {cat:32s} {n:5d}")

    if not entries:
        raise RuntimeError(
            "No usable library entries after resampling. Check the splib07 "
            "directory layout and target wavelengths."
        )

    # Persist the cache for next time. Only after successful build, so we
    # never write a partial cache.
    if cache_path is not None:
        try:
            _save_cache(cache_path, entries)
            if verbose:
                print(f"      cached to {cache_path}.npz")
        except OSError as e:
            if verbose:
                print(f"      WARNING: failed to write cache: {e}")
    return entries


def stack_library(entries: List[LibraryEntry]) -> np.ndarray:
    """Stack a list of LibraryEntry into an (N, n_bands) float32 array."""
    return np.stack([e.refl for e in entries], axis=0)
