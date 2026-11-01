"""
Planet Tanager delivered as ENVI: a `.hdr` plus its raw data file.

The second input format of the native Tanager sensor. TanagerHelper
(tanager_helper.py) dispatches here when the file it is handed is a `.hdr`;
read_tanager_metadata, the georef resolver and scene onboarding all reach
this module through it, so the HDF5 and ENVI paths share one band table,
one validity rule and one scene id scheme.

Why it exists: users re-export Tanager from ENVI-based tools (ENVI, SNAP,
ISOFIT) and upload `.hdr` + `.img`. Without this they would have to go in
dressed as AVIRIS-NG, whose reader hand-builds memmaps and silently misreads
BSQ. Here GDAL (through rasterio) reads the data, so BSQ, BIL and BIP all
come out as the same (C, H, W) cube, and `map info` / `coordinate system
string` become the CRS and affine the export needs.

What the ENVI path does not have: Planet's nodata, cloud and cirrus layers.
Validity is then value != `data ignore value` AND finite AND band good,
which is as strict as the HDF5 path minus the masks; the builder skips
`quality_masks_to_apply` with a warning.

Gotchas:
  - `reflectance scale factor` divides the data into 0-1 (ENVI's meaning).
    Without it the scale is detected from a decimated sample; which one was
    used is recorded in `TanagerMetadata.reflectance_scale_source`.
  - `wavelength units` may be micrometres; wavelengths and FWHM are
    converted to nm. With no unit, values below 100 are taken as um.
  - No `data ignore value` means fill is NaN, so only the finite test bites.
  - rasterio band indexes are 1-based file order; the band table is sorted.
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from app.models.file_processing.tanager_metadata import TanagerGrid, TanagerMetadata
from app.utils.files.envi_helper import _csv_floats, _parse_envi_header

logger = logging.getLogger("TanagerHelper")

ENVI_DATA_SUFFIXES = (".img", ".bsq", ".bil", ".bip", ".dat", "")

# A Planet item id anywhere in a name or description: YYYYMMDD_HHMMSS_xx_yyyy.
_ITEM_ID_RE = re.compile(r"(?<![0-9A-Za-z])(\d{8}_\d{6}_[0-9A-Za-z]{2}_[0-9A-Za-z]{4})(?![0-9A-Za-z])")

# Sample p99 above this is integer-scaled (0-10000) reflectance. 0-1 data stays
# below ~1.5 even over snow or glint; scaled data sits in the hundreds.
_SCALED_P99_THRESHOLD = 10.0
_DETECTED_SCALE = 10000.0
# Decimated read used for that detection: a few bands, at most this size.
_SAMPLE_BANDS = 8
_SAMPLE_PIXELS = 256

_UM_UNITS = ("micrometers", "micrometer", "microns", "micron", "um", "µm")
_NM_UNITS = ("nanometers", "nanometer", "nm")


def _open(path):
    """rasterio.open without the NotGeoreferencedWarning an ENVI file lacking map info raises."""
    import warnings

    import rasterio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
        return rasterio.open(path)


def envi_data_file(hdr: Path) -> Optional[Path]:
    """The raw data file beside `hdr`: `<stem>.img|.bsq|.bil|.bip|.dat|<stem>`, or `<data>.hdr`'s `<data>`."""
    hdr = Path(hdr)
    stem = hdr.parent / hdr.stem
    if stem.suffix.lower() in ENVI_DATA_SUFFIXES and stem.is_file():
        return stem                                        # scene.img + scene.img.hdr
    for ext in ENVI_DATA_SUFFIXES:
        for cand in (Path(f"{stem}{ext}"), Path(f"{stem}{ext.upper()}")):
            if cand.is_file() and cand != hdr:
                return cand
    return None


def item_id_in(text: Optional[str]) -> Optional[str]:
    m = _ITEM_ID_RE.search(text or "")
    return m.group(1) if m else None


def _to_nm_factor(units: Optional[str], wavelengths: np.ndarray) -> float:
    u = (units or "").strip().lower()
    if u in _UM_UNITS:
        return 1000.0
    if u in _NM_UNITS:
        return 1.0
    guess = 1000.0 if np.nanmax(wavelengths) < 100.0 else 1.0
    logger.warning("ENVI wavelength units %r not recognised; read as %s", units, "um" if guess > 1 else "nm")
    return guess


def _acquisition(scene_id: str, tokens: dict) -> Optional[datetime]:
    from app.utils.files.tanager_helper import _acquisition_time

    at = _acquisition_time(scene_id)
    raw = tokens.get("acquisition time")
    if at is None and raw:
        try:
            at = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            at = at if at.tzinfo else at.replace(tzinfo=timezone.utc)
        except ValueError:
            logger.warning("ENVI acquisition time %r not ISO 8601; ignored", raw)
    return at


def _grid_from(src) -> Optional[TanagerGrid]:
    """TanagerGrid from GDAL's reading of `map info`; None when there is no usable north-up EPSG grid."""
    if src.crs is None:
        return None
    t = src.transform
    epsg = src.crs.to_epsg()
    if t.b != 0 or t.d != 0 or epsg is None:
        logger.warning("ENVI grid is rotated or its CRS has no EPSG code; no scene bbox (export still georeferences)")
        return None
    return TanagerGrid(
        x_dim=src.width, y_dim=src.height,
        upper_left_m=(t.c, t.f), lower_right_m=(t.c + t.a * src.width, t.f + t.e * src.height),
        projection="ENVI map info", epsg=epsg,
    )


def _detect_scale(src, fill: float, good_file_idx: np.ndarray) -> Tuple[float, str]:
    from rasterio.enums import Resampling

    pick = good_file_idx[np.linspace(0, good_file_idx.size - 1, min(_SAMPLE_BANDS, good_file_idx.size)).astype(int)]
    shape = (pick.size, min(src.height, _SAMPLE_PIXELS), min(src.width, _SAMPLE_PIXELS))
    arr = src.read(indexes=[int(i) + 1 for i in pick], out_shape=shape, out_dtype="float32",
                   resampling=Resampling.nearest)
    vals = arr[(arr != fill) & np.isfinite(arr)]
    if vals.size == 0:
        return 1.0, "assumed_0_1_no_valid_sample"
    if float(np.percentile(vals, 99)) > _SCALED_P99_THRESHOLD:
        return _DETECTED_SCALE, "detected_0_10000"
    return 1.0, "detected_0_1"


def read_tanager_envi_metadata(hdr: str | Path) -> TanagerMetadata:
    """Band table, grid, scale and identity of an ENVI Tanager scene; reads only a decimated sample."""
    from app.utils.files.tanager_helper import _sorted_band_table

    hdr = Path(hdr)
    data = envi_data_file(hdr)
    if data is None:
        raise ValueError(f"ENVI header {hdr.name} has no data file beside it ({', '.join(ENVI_DATA_SUFFIXES[:-1])} or none)")
    tokens = _parse_envi_header(hdr.read_text(errors="ignore"))
    if "wavelength" not in tokens:
        raise ValueError(f"ENVI header {hdr.name} has no wavelength list")
    bands = int(tokens["bands"])
    wl_raw = np.asarray(_csv_floats(tokens["wavelength"]), dtype=np.float64)
    if wl_raw.size != bands:
        raise ValueError(f"{hdr.name}: {wl_raw.size} wavelengths for {bands} bands")
    to_nm = _to_nm_factor(tokens.get("wavelength units"), wl_raw)
    wl = wl_raw * to_nm

    fwhm_derived = "fwhm" not in tokens
    if fwhm_derived:
        order = np.argsort(wl, kind="stable")
        fw = np.empty_like(wl)
        fw[order] = np.gradient(wl[order])
        logger.info("ENVI header %s has no fwhm; derived from band spacing", hdr.name)
    else:
        fw = np.asarray(_csv_floats(tokens["fwhm"])[:bands], dtype=np.float64) * to_nm
    bbl = np.asarray(_csv_floats(tokens["bbl"]), dtype=np.float64) if "bbl" in tokens else np.ones(bands)
    if bbl.size != bands:
        raise ValueError(f"{hdr.name}: bbl has {bbl.size} entries for {bands} bands")
    order, wl_s, fw_s, good = _sorted_band_table(wl, fw, bbl > 0)
    fill = float(tokens["data ignore value"]) if "data ignore value" in tokens else float("nan")

    with _open(data) as src:
        if src.count != bands:
            raise ValueError(f"{data.name}: GDAL reads {src.count} bands, header says {bands}")
        grid = _grid_from(src)
        height, width = src.height, src.width
        if "reflectance scale factor" in tokens:
            scale, scale_src = float(tokens["reflectance scale factor"]), "header"
        else:
            good_file = np.asarray(order)[np.asarray(good, bool)]
            scale, scale_src = _detect_scale(src, fill, good_file if good_file.size else np.asarray(order))
    names = [n.strip() for n in tokens["band names"].split(",")] if "band names" in tokens else None
    description = tokens.get("description")
    scene_id = item_id_in(hdr.name) or item_id_in(description) or hdr.stem
    logger.info("ENVI Tanager %s: %d bands, %s, scale %g (%s), fwhm %s", scene_id, bands,
                tokens.get("interleave", "?"), scale, scale_src, "derived" if fwhm_derived else "from header")
    return TanagerMetadata(
        source_path=str(hdr), scene_id=scene_id, strip_id=None, acquisition_at=_acquisition(scene_id, tokens),
        bands=bands, height=height, width=width, fill_value=fill,
        wavelengths_nm=wl_s, fwhm_nm=fw_s, good_band=good, file_band_index=order, grid=grid,
        file_format="envi", data_path=str(data), interleave=tokens.get("interleave", "").strip().lower() or None,
        reflectance_scale_factor=scale, reflectance_scale_source=scale_src, fwhm_derived=fwhm_derived,
        band_names=[names[i] for i in order] if names and len(names) == bands else None,
    )


def read_envi_bands(md: TanagerMetadata, file_idx: np.ndarray) -> np.ndarray:
    """(C, H, W) float32 for 0-based file band indexes, scaled to 0-1; fill voxels keep the fill value.

    One GDAL read for all bands: per-band reads of a BIP file would scan it once per band.
    """
    with _open(md.data_path) as src:
        out = src.read(indexes=[int(i) + 1 for i in file_idx], out_dtype="float32")
    if md.reflectance_scale_factor != 1.0:
        fill = out == md.fill_value
        out /= np.float32(md.reflectance_scale_factor)
        out[fill] = md.fill_value
    return out
