"""One-shot USGS splib07 curation script.

Reads the raw 6.7 GB splib07 distribution at a user-supplied path
and produces a slim ~40 MB on-disk bundle with the only pieces our
spectral-match action needs:

  - the ASD-native wavelength axis (350-2500 nm, 2151 channels)
  - the three ASD FWHM files (FR / HR / NG sub-instruments)
  - one ASCII file per ASD-instrument spectrum, organised by chapter
  - an `index.json` that maps each spectrum to a pretty name,
    chapter, ASD sub-instrument, and source-record number

What's deliberately skipped (and why):

  - Pre-resampled copies for other instruments (`_cvAVIRIS*`, `_cvHyperion`,
    `_cvLandsat`, ...). We do our own resampling per Allotrope sensor at
    cache-build time, so a second-hand resample would compound error.
  - BECK / NIC4 spectra. Their wavelength ranges don't overlap with
    any of our hyperspectral sensors (200-3000 nm Beckman; 1.12-216 um
    Nicolet mid-IR). Including them would just produce no-coverage
    NaNs after resampling.
  - The binary SPECPR copy, GIFplots/, HTMLmetadata/, PDFs. Pure
    documentation overhead for our runtime.

The script is content-addressed: any change to which files are included
or how they're parsed should bump --version, which the cache-build
CLI's SHA-256 key (later) keys on. That way changing curation logic
invalidates the per-sensor caches automatically.

Usage:

    python scripts/curate_splib07.py \\
        --raw  data/splib/usgs_splib07 \\
        --out  data/splib07_slim \\
        --version 1

The output tree is the source-of-truth for the Allotrope spectral-match
action. We mount this folder into the worker container via a Docker
volume named ``allotrope_splib07``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


logger = logging.getLogger("curate_splib07")


# Filenames inside ASCIIdata_splib07a/. Hardcoded because splib07 has
# been stable since release — these names won't change without a
# version bump on USGS's side, which we'd catch in --version anyway.
_ASD_WAVELENGTHS_FILENAME = (
    "splib07a_Wavelengths_ASD_0.35-2.5_microns_2151_ch.txt"
)
_ASD_FWHM_FILENAMES = {
    # ASD subtype tag (as it appears in spectrum filenames) -> FWHM file
    "ASDFR": "splib07a_Bandpass_(FWHM)_ASDFR_StandardResolution.txt",
    "ASDHR": "splib07a_Bandpass_(FWHM)_ASDHR_High-Resolution.txt",
    "ASDNG": "splib07a_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt",
}

# Chapter folder name on disk -> short slug we use in the slim bundle.
_CHAPTER_SLUGS = {
    "ChapterA_ArtificialMaterials": "artificial",
    "ChapterC_Coatings":             "coatings",
    "ChapterL_Liquids":              "liquids",
    "ChapterM_Minerals":             "minerals",
    "ChapterO_OrganicCompounds":     "organics",
    "ChapterS_SoilsAndMixtures":     "soils",
    "ChapterV_Vegetation":           "vegetation",
}

# Filename pattern for an ASD-instrument spectrum file. Tail "a" / "b"
# after the subtype is a measurement-batch tag; we keep both.
# Examples:
#   splib07a_Actinolite_HS116.1B_ASDFRb_AREF.txt
#   splib07a_Calcite_WS272_ASDNGb_AREF.txt
_ASD_FILENAME_RE = re.compile(
    r"^splib07a_(?P<material>.+?)_(?P<sample>[A-Z0-9._]+?)_"
    r"(?P<subtype>ASD(?:FR|HR|NG))(?P<batch>[a-z]?)_"
    r"(?P<product>[A-Z]+)\.txt$"
)


@dataclass
class _SpectrumRef:
    """One library entry we plan to copy into the slim bundle."""

    src_path: Path
    chapter_slug: str
    asd_subtype: str  # ASDFR | ASDHR | ASDNG
    material: str
    sample: str
    record: int  # parsed from the file header line
    pretty_name: str  # parsed from the file header line


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def _parse_header(line: str) -> tuple[int, str]:
    """Pull the record number + pretty name out of a splib07 header.

    Example header line:
        ' splib07a Record=108: Actinolite HS116.1B          ASDFRb AREF'

    Returns (108, 'Actinolite HS116.1B').
    """
    # The header is stable across all splib07a files. Strip leading space,
    # split at "Record=" + ": ", then drop the trailing instrument/product tag.
    s = line.strip()
    if "Record=" not in s or ":" not in s:
        raise ValueError(f"unrecognised splib07 header: {line!r}")
    _, rest = s.split("Record=", 1)
    rec_str, name_part = rest.split(":", 1)
    record = int(rec_str.strip())
    # Pretty name = everything before the instrument tag at the end.
    # Instrument tags are 3-6 uppercase chars at the end of the line:
    # 'ASDFRb', 'ASDHRa', 'BECKa', 'NIC4', etc. Just trim trailing
    # whitespace + drop the last two whitespace-separated tokens (instr,
    # product) — those are always present.
    tokens = name_part.strip().split()
    if len(tokens) >= 2:
        pretty = " ".join(tokens[:-2]).strip()
    else:
        pretty = " ".join(tokens).strip()
    return record, pretty


# ---------------------------------------------------------------------------
# Walking the raw tree
# ---------------------------------------------------------------------------


def _discover_spectra(splib07a_root: Path) -> list[_SpectrumRef]:
    """Walk every chapter folder, keep ASD entries, parse headers."""
    refs: list[_SpectrumRef] = []
    for chapter_dirname, slug in _CHAPTER_SLUGS.items():
        chapter_dir = splib07a_root / chapter_dirname
        if not chapter_dir.is_dir():
            logger.warning("chapter folder missing: %s", chapter_dir)
            continue
        kept = 0
        skipped = 0
        for path in sorted(chapter_dir.iterdir()):
            if not path.is_file() or path.suffix != ".txt":
                continue
            m = _ASD_FILENAME_RE.match(path.name)
            if not m:
                # Non-ASD file (BECK, NIC4, AVIRIS, ...). Skipped silently.
                skipped += 1
                continue
            try:
                with path.open() as fh:
                    header = fh.readline()
                record, pretty = _parse_header(header)
            except Exception as exc:  # noqa: BLE001
                logger.warning("skipping %s: header parse failed: %s", path, exc)
                continue
            refs.append(
                _SpectrumRef(
                    src_path=path,
                    chapter_slug=slug,
                    asd_subtype=m.group("subtype"),
                    material=m.group("material"),
                    sample=m.group("sample"),
                    record=record,
                    pretty_name=pretty,
                )
            )
            kept += 1
        logger.info(
            "chapter %-22s kept=%4d  skipped=%4d (non-ASD)",
            chapter_dirname, kept, skipped,
        )
    return refs


# ---------------------------------------------------------------------------
# Wavelength + FWHM conversion (microns -> nm)
# ---------------------------------------------------------------------------


def _read_numeric_column_file(
    path: Path,
    expected_count: Optional[int] = None,
) -> list[float]:
    """Read a splib07 single-column ASCII file (header + N floats).

    The first line is the human-readable header (record number + label).
    Subsequent lines are one float each. We skip the header and parse
    the rest.

    splib07 stores wavelengths and FWHM in MICRONS; the caller converts.
    """
    values: list[float] = []
    with path.open() as fh:
        next(fh)  # header
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            values.append(float(raw))
    if expected_count is not None and len(values) != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} values, got {len(values)}"
        )
    return values


def _to_nm(values_microns: Iterable[float]) -> list[float]:
    """Microns -> nm. 0.35 um -> 350 nm. Trivial but worth being explicit."""
    return [v * 1000.0 for v in values_microns]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_axis_file(
    values_nm: list[float],
    out_path: Path,
    header: str,
) -> None:
    """Write a one-value-per-line ASCII file with a one-line header.

    Mirrors the splib07 file shape so downstream consumers feel
    familiar, but the values are in nanometers and the header is
    human-friendly (we're not stuck with the splib07 wording).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(header.rstrip() + "\n")
        for v in values_nm:
            fh.write(f"{v:.6f}\n")


def _copy_spectrum(
    ref: _SpectrumRef,
    spectra_dir: Path,
) -> tuple[Path, int]:
    """Copy one spectrum file into the slim bundle. Returns the relative
    path stored in index.json + the value count (lines after header)."""
    dst_dir = spectra_dir / ref.chapter_slug
    dst_dir.mkdir(parents=True, exist_ok=True)
    # Keep the source filename verbatim — splib07 record numbers + sample
    # IDs make every filename globally unique already.
    dst = dst_dir / ref.src_path.name
    shutil.copy2(ref.src_path, dst)
    # Count values for the index (sanity check on cache-build later).
    n = 0
    with dst.open() as fh:
        for i, _ in enumerate(fh):
            if i == 0:
                continue  # header
            n += 1
    return dst, n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Curate the raw USGS splib07 distribution down to the slim "
            "ASD-only ASCII bundle the Allotrope spectral-match action "
            "consumes."
        ),
    )
    p.add_argument(
        "--raw",
        type=Path,
        required=True,
        help=(
            "Path to the raw splib07 root (the folder containing "
            "ASCIIdata/, SPECPRsplib07/, etc.). Example: "
            "data/splib/usgs_splib07"
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output directory for the slim bundle. Will be wiped + "
            "recreated. Example: data/splib07_slim"
        ),
    )
    p.add_argument(
        "--version",
        type=str,
        required=True,
        help=(
            "Curation version tag (free-form string). Stored in "
            "index.json + becomes part of the per-sensor cache key "
            "downstream so bumping it invalidates caches."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-chapter / per-file logging.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s — %(message)s",
    )

    raw_root: Path = args.raw.resolve()
    out_root: Path = args.out.resolve()
    splib07a = raw_root / "ASCIIdata" / "ASCIIdata_splib07a"
    if not splib07a.is_dir():
        logger.error(
            "expected %s under --raw; got: %s",
            "ASCIIdata/ASCIIdata_splib07a/", splib07a,
        )
        return 2

    # --- Output reset ----------------------------------------------------
    if out_root.exists():
        logger.info("wiping existing output dir: %s", out_root)
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # --- Wavelengths -----------------------------------------------------
    wl_src = splib07a / _ASD_WAVELENGTHS_FILENAME
    if not wl_src.is_file():
        logger.error("missing ASD wavelengths file: %s", wl_src)
        return 2
    wl_um = _read_numeric_column_file(wl_src, expected_count=2151)
    wl_nm = _to_nm(wl_um)
    _write_axis_file(
        wl_nm,
        out_root / "wavelengths_asd_nm.txt",
        header="# ASD wavelength axis, 350.0 - 2500.0 nm, 2151 channels (nm)",
    )
    logger.info(
        "wavelengths: %d channels, %.1f to %.1f nm",
        len(wl_nm), wl_nm[0], wl_nm[-1],
    )

    # --- FWHM files ------------------------------------------------------
    fwhm_dir = out_root / "fwhm"
    fwhm_dir.mkdir()
    for subtype, fname in _ASD_FWHM_FILENAMES.items():
        src = splib07a / fname
        if not src.is_file():
            logger.error("missing FWHM file for %s: %s", subtype, src)
            return 2
        vals_um = _read_numeric_column_file(src, expected_count=2151)
        vals_nm = _to_nm(vals_um)
        _write_axis_file(
            vals_nm,
            fwhm_dir / f"{subtype}.txt",
            header=f"# splib07 ASD FWHM for {subtype}, 2151 channels (nm)",
        )
    logger.info("FWHM: wrote FR / HR / NG (all 2151-channel)")

    # --- Discover spectra ------------------------------------------------
    refs = _discover_spectra(splib07a)
    if not refs:
        logger.error("no ASD spectra discovered under %s", splib07a)
        return 2
    logger.info("total ASD spectra to copy: %d", len(refs))

    # --- Copy + build index ---------------------------------------------
    spectra_dir = out_root / "spectra"
    index_entries: list[dict] = []
    per_subtype_counts: dict[str, int] = {}
    per_chapter_counts: dict[str, int] = {}
    for ref in refs:
        dst, value_count = _copy_spectrum(ref, spectra_dir)
        rel = dst.relative_to(out_root).as_posix()
        index_entries.append({
            "name": ref.pretty_name,
            "material": ref.material,
            "sample": ref.sample,
            "chapter": ref.chapter_slug,
            "asd_subtype": ref.asd_subtype,
            "record": ref.record,
            "value_count": value_count,
            "path": rel,
        })
        per_subtype_counts[ref.asd_subtype] = (
            per_subtype_counts.get(ref.asd_subtype, 0) + 1
        )
        per_chapter_counts[ref.chapter_slug] = (
            per_chapter_counts.get(ref.chapter_slug, 0) + 1
        )

    # --- Index file ------------------------------------------------------
    index_payload = {
        "version": args.version,
        "source_root": str(raw_root),
        "wavelengths_file": "wavelengths_asd_nm.txt",
        "fwhm_files": {
            subtype: f"fwhm/{subtype}.txt" for subtype in _ASD_FWHM_FILENAMES
        },
        "wavelength_unit": "nm",
        "fwhm_unit": "nm",
        "n_entries": len(index_entries),
        "n_channels": 2151,
        "per_chapter_counts": per_chapter_counts,
        "per_subtype_counts": per_subtype_counts,
        "entries": index_entries,
    }
    (out_root / "index.json").write_text(
        json.dumps(index_payload, indent=2)
    )

    # --- Summary ---------------------------------------------------------
    logger.info("done. wrote %d entries to %s", len(index_entries), out_root)
    logger.info(
        "per-chapter counts: %s",
        json.dumps(per_chapter_counts, sort_keys=True),
    )
    logger.info(
        "per-subtype counts: %s",
        json.dumps(per_subtype_counts, sort_keys=True),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
