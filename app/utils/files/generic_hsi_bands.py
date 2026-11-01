"""
Where a hyperspectral file keeps its band centres and widths, tried in a fixed order.

Used by app/utils/files/generic_hsi_helper.py (`sensor_type="generic_hsi"`).
Every source returns the table in FILE band order; the helper sorts it.

Why it exists: the evaluation data arrives in whatever shape the provider
chose, and a cube whose wavelengths cannot be found cannot be scored, matched
or even band-filtered. So the reader tries every place we have seen a band
table, first hit wins (docs/proposals/robust-hsi-ingestion.md):

  1. per-band GDAL metadata: `wavelength`, the misspelt `wavelenght_units`
     GHOSt's GeoTIFFs carry, Tanager-style `wavelengths`/`fwhm`, NetCDF
     `NETCDF_DIM_<wave>`, and GDAL's IMAGERY domain (CENTRAL_WAVELENGTH_UM);
  2. the ENVI header (`wavelength`, `fwhm`, `wavelength units`, `bbl`);
  3. HDF5 / NetCDF attributes: GDAL's flattened root tags, then h5py on the
     dataset's own attributes and 1-D `wavelength*` / `fwhm*` datasets;
  4. band descriptions such as `Band 1 (396.8 nm)` - the number in the
     parentheses, never the band label (GHOSt dataset 3 has position 218
     labelled `Band 224`);
  5. a wavelength file (CSV / TXT / JSON) uploaded beside the cube.

Gotchas:
  - Units: micrometres become nm. With no unit, values all below 100 are
    taken as micrometres; both cases are recorded in `notes`.
  - A source only counts when it gives exactly one finite wavelength per band.
  - FWHM is taken from the first source that has an exact full list, which
    need not be the wavelength source (GDAL's ENVI tags carry wavelength, and
    FWHM only in the IMAGERY domain rounded to 1 nm; the header is exact).
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_UM_UNITS = {"micrometers", "micrometer", "micrometre", "micrometres", "microns", "micron", "um", "µm", "μm"}
_NM_UNITS = {"nanometers", "nanometer", "nanometre", "nanometres", "nm"}

# Per-band tag keys (lower-cased) that hold a band centre / width / good flag.
_WL_KEYS = ("wavelength", "wavelengths", "center_wavelength", "central_wavelength", "wavelength_nm", "band_wavelength")
_FWHM_KEYS = ("fwhm", "fwhm_nm", "bandwidth")
_GOOD_KEYS = ("good_wavelength", "good_wavelengths", "bbl")

# "Band 1 (396.8 nm)", "Band 472 (2507.2)", "396.8 nm".
_DESC_PAREN_RE = re.compile(r"\(\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*([a-zA-Zµμ]*)\s*\)")
_DESC_BARE_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\s*(nm|um|µm|μm|nanometers?|micrometers?)\b", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

SIDECAR_SUFFIXES = (".csv", ".txt", ".json")
# The operator's ingest options (scale override) live in this sidecar, never a wavelength file.
INGEST_OPTIONS_NAME = "allotrope_ingest.json"


@dataclass
class BandTable:
    """One source's band table, FILE band order, nm."""

    wavelengths: np.ndarray
    source: str
    fwhm: Optional[np.ndarray] = None
    fwhm_source: Optional[str] = None
    good: Optional[np.ndarray] = None
    units_read: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    # False for GDAL's IMAGERY-domain FWHM_UM, printed to 0.001 um (8.7 nm reads as 9): a later
    # source with an exact list (the ENVI header) is preferred.
    fwhm_exact: bool = True


def to_nm_factor(units: Optional[str], values: np.ndarray) -> Tuple[float, Optional[str]]:
    """(factor to nm, note). Unknown units are guessed from magnitude, and the guess is noted."""
    u = (units or "").strip().strip("{}").strip().lower()
    if u in _UM_UNITS or "micro" in u:
        return 1000.0, f"wavelength units {units!r}: micrometres converted to nm"
    if u in _NM_UNITS or "nano" in u:
        return 1.0, None
    guess_um = bool(values.size) and float(np.nanmax(values)) < 100.0
    return (1000.0 if guess_um else 1.0), (
        f"wavelength units {units!r} not stated; values {'< 100 read as micrometres' if guess_um else 'read as nm'}")


def _floats(text: str) -> List[float]:
    return [float(x) for x in _NUM_RE.findall(text or "")]


def _full(values: Sequence[float], count: int) -> Optional[np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    return arr if arr.size == count and np.all(np.isfinite(arr)) else None


def _finish(wl: np.ndarray, units: Optional[str], source: str, fwhm=None, fwhm_units=None, good=None) -> BandTable:
    factor, note = to_nm_factor(units, wl)
    table = BandTable(wavelengths=wl * factor, source=source, units_read=units, good=good)
    if note:
        table.notes.append(note)
    if fwhm is not None:
        f_factor = to_nm_factor(fwhm_units, fwhm)[0] if fwhm_units else factor
        table.fwhm, table.fwhm_source = fwhm * f_factor, source
    return table


# --- 1. per-band metadata -------------------------------------------------

def from_band_tags(src) -> Optional[BandTable]:
    count = src.count
    wl, fw, good, units, fw_units, fw_exact = [], [], [], None, None, True
    for b in range(1, count + 1):
        tags = {k.lower(): v for k, v in src.tags(b).items()}
        try:
            imagery = {k.upper(): v for k, v in src.tags(b, ns="IMAGERY").items()}
        except Exception:   # noqa: BLE001 - a driver without the domain
            imagery = {}
        value = next((tags[k] for k in _WL_KEYS if k in tags), None)
        if value is None:
            value = next((v for k, v in tags.items() if k.startswith("netcdf_dim_") and ("wave" in k or "lambda" in k)), None)
        if value is None and "CENTRAL_WAVELENGTH_UM" in imagery:
            value, units = imagery["CENTRAL_WAVELENGTH_UM"], "um"
        nums = _floats(value) if value is not None else []
        if len(nums) != 1:
            return None
        wl.append(nums[0])
        units = units or next((v for k, v in tags.items() if k.startswith("wavel") and "unit" in k), None)
        f = next((tags[k] for k in _FWHM_KEYS if k in tags), None)
        if f is None and "FWHM_UM" in imagery:
            f, fw_units, fw_exact = imagery["FWHM_UM"], "um", False
        fw_units = fw_units or next((v for k, v in tags.items() if k.startswith("fwhm") and "unit" in k), None)
        fw.append(_floats(f)[0] if f is not None and _floats(f) else np.nan)
        g = next((tags[k] for k in _GOOD_KEYS if k in tags), None)
        good.append(_floats(g)[0] if g is not None and _floats(g) else 1.0)
    wl_arr = _full(wl, count)
    if wl_arr is None:
        return None
    table = _finish(wl_arr, units, "band_metadata", fwhm=_full(fw, count), fwhm_units=fw_units,
                    good=np.asarray(good) > 0)
    table.fwhm_exact = fw_exact
    return table


# --- 2. ENVI header -------------------------------------------------------

def from_envi_header(tokens: Optional[Dict[str, str]], count: int) -> Optional[BandTable]:
    if not tokens or "wavelength" not in tokens:
        return None
    wl = _full(_floats(tokens["wavelength"]), count)
    if wl is None:
        return None
    fw = _full(_floats(tokens["fwhm"]), count) if "fwhm" in tokens else None
    bbl = _full(_floats(tokens["bbl"]), count) if "bbl" in tokens else None
    return _finish(wl, tokens.get("wavelength units"), "envi_header", fwhm=fw,
                   good=(bbl > 0) if bbl is not None else None)


# --- 3. HDF5 / NetCDF attributes -----------------------------------------

def from_container_tags(root_tags: Dict[str, str], variable: Optional[str], count: int) -> Optional[BandTable]:
    """GDAL's flattened HDF5 root tags (`<path>_<var>_wavelengths`) and NetCDF `NETCDF_DIM_<dim>_VALUES`."""
    lower = {k.lower(): (k, v) for k, v in root_tags.items()}
    var = (variable or "").lower()

    def pick(suffixes):
        hits = [(k, v) for k, (_, v) in lower.items()
                if k.endswith(suffixes) and "good" not in k and "bad" not in k
                and _full(_floats(v), count) is not None]
        hits.sort(key=lambda kv: (var not in kv[0], len(kv[0])))
        return hits[0] if hits else (None, None)

    key, value = pick(("_wavelength", "_wavelengths", "wavelength", "wavelengths"))
    if key is None:
        key, value = next(((k, v) for k, (_, v) in lower.items() if k.startswith("netcdf_dim_") and k.endswith("_values")
                           and ("wave" in k or "lambda" in k) and _full(_floats(v), count) is not None), (None, None))
        if key is None:
            return None
        dim = key[len("netcdf_dim_"):-len("_values")]
        units = next((v for k, (_, v) in lower.items() if k == f"{dim}#units"), None)
    else:
        units = next((v for k, (_, v) in lower.items() if k in (key + "_units", key + "#units")), None)
    fkey, fvalue = pick(("_fwhm", "fwhm"))
    return _finish(_full(_floats(value), count), units, "hdf_attributes",
                   fwhm=_full(_floats(fvalue), count) if fkey else None)


def from_h5py(container: Path, variable_path: Optional[str], count: int) -> Optional[BandTable]:
    """h5py look at the chosen dataset's attributes and at 1-D `wavelength*` / `fwhm*` datasets of length `count`."""
    try:
        import h5py
    except ImportError:
        return None
    want = (variable_path or "").strip("/").replace(" ", "_").lower()
    found: Dict[str, Tuple[np.ndarray, Optional[str]]] = {}

    def units_of(obj) -> Optional[str]:
        for k in ("units", "unit", "Units", "Unit"):
            if k in obj.attrs:
                v = obj.attrs[k]
                return v.decode() if isinstance(v, bytes) else str(v)
        return None

    try:
        with h5py.File(container, "r") as f:
            def visit(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if name.replace(" ", "_").lower() == want:
                    for k, v in obj.attrs.items():
                        kl = k.lower()
                        arr = np.asarray(v).reshape(-1)
                        if arr.dtype.kind in "fiu" and arr.size == count:
                            if "wavel" in kl and "unit" not in kl:
                                found.setdefault("wl", (arr.astype(float), None))
                            elif "fwhm" in kl:
                                found.setdefault("fwhm", (arr.astype(float), None))
                elif obj.ndim == 1 and obj.shape[0] == count and obj.dtype.kind in "fiu":
                    leaf = name.rsplit("/", 1)[-1].lower()
                    if "wavel" in leaf or leaf in ("lambda", "band_centers"):
                        found.setdefault("wl", (obj[()].astype(float), units_of(obj)))
                    elif "fwhm" in leaf:
                        found.setdefault("fwhm", (obj[()].astype(float), units_of(obj)))
            f.visititems(visit)
    except OSError:
        return None
    if "wl" not in found or _full(found["wl"][0], count) is None:
        return None
    fw = found.get("fwhm")
    return _finish(found["wl"][0], found["wl"][1], "hdf_attributes",
                   fwhm=_full(fw[0], count) if fw else None, fwhm_units=fw[1] if fw else None)


# --- 4. band descriptions -------------------------------------------------

def from_descriptions(descriptions: Sequence[Optional[str]], count: int) -> Optional[BandTable]:
    wl, units = [], set()
    for d in descriptions:
        m = _DESC_PAREN_RE.search(d or "") or _DESC_BARE_RE.search(d or "")
        if m is None:
            return None
        wl.append(float(m.group(1)))
        if m.group(2):
            units.add(m.group(2).lower())
    arr = _full(wl, count)
    if arr is None:
        return None
    return _finish(arr, units.pop() if len(units) == 1 else None, "band_descriptions")


# --- 5. an uploaded wavelength file ----------------------------------------

def _parse_wavelength_file(path: Path) -> Tuple[List[float], Optional[List[float]], Optional[str]]:
    text = path.read_text(errors="ignore")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [float(v) for v in obj], None, None
        lower = {k.lower(): v for k, v in obj.items()}
        wl = lower.get("wavelengths", lower.get("wavelength"))
        if wl is None:
            raise ValueError("no 'wavelength' list")
        fw = lower.get("fwhm")
        units = lower.get("wavelength_units", lower.get("units"))
        return [float(v) for v in wl], ([float(v) for v in fw] if fw else None), units
    rows, header = [], ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            header += " " + line
            continue
        nums = _floats(line)
        if not nums or not re.match(r"^[-+]?\d", line):
            header += " " + line
            continue
        rows.append(nums)
    units = "um" if re.search(r"\b(um|µm|micro\w*)\b", header, re.IGNORECASE) else (
        "nm" if re.search(r"\b(nm|nano\w*)\b", header, re.IGNORECASE) else None)
    if len(rows) == 1:
        return rows[0], None, units
    width = min(len(r) for r in rows)
    cols = [[r[i] for r in rows] for i in range(width)]
    # An ENVI-style "band, wavelength, fwhm" table: drop the leading 1..N / 0..N-1 index.
    n = len(rows)
    if width >= 2 and (cols[0] == list(range(1, n + 1)) or cols[0] == list(range(n))):
        cols = cols[1:]
    return cols[0], (cols[1] if len(cols) > 1 else None), units


def from_sidecar(paths: Sequence[Path], count: int) -> Optional[BandTable]:
    """The first CSV / TXT / JSON in the upload that holds exactly `count` wavelengths."""
    cands = [p for p in paths if p.suffix.lower() in SIDECAR_SUFFIXES and p.name.lower() != INGEST_OPTIONS_NAME]
    cands.sort(key=lambda p: (not re.search(r"wave|wl|band|lambda", p.name, re.IGNORECASE), p.name))
    for p in cands:
        try:
            wl, fw, units = _parse_wavelength_file(p)
        except (ValueError, json.JSONDecodeError, AttributeError, TypeError):
            continue
        arr = _full(wl, count)
        if arr is not None:
            table = _finish(arr, units, f"wavelength_file:{p.name}", fwhm=_full(fw, count) if fw else None)
            return table
    return None
