"""
Map coordinates of a generic_hsi scene: from the file, from an OSK `.igm` geometry file, or none.

Used twice with the same answer: by GenericHsiHelper at onboarding (the
pre-flight report's georef line, the Scene bbox and native_projection) and
by the export path through app/georef/from_scene_file.py
(`resolve_generic_hsi_georef`), which turns GeorefUnavailable into 422
crs_missing.

Why it exists: the vendable holds no spatial reference, so a submission's
GeoTIFF gets its grid from here. Three cases, in this order:
  1. the file's own CRS + affine (GeoTIFF, ENVI `map info`, HDF-EOS grid).
     A geographic CRS (EPSG:4326, as GHOSt datasets 1 and 3) is a valid grid;
  2. an unrectified OSK product with a `.igm` beside it - per-pixel latitude
     and longitude. Handled like PRISMA's lat/lon fallback: an EPSG:4326
     affine fitted to the swath's bounding box. That is approximate (a
     rotated swath is not an axis-aligned grid), and the report says so;
  3. none: onboarding and scoring still run, but export refuses with
     NO_MAP_MESSAGE. A map that cannot be placed must never be submitted.

Gotchas:
  - The IGM is read with numpy from its own ENVI header, not with GDAL.
    GDAL looks for `<stem>.hdr` before `<file>.hdr`, so opening `scene.igm`
    beside `scene.hdr` would pair the geometry with the CUBE's header.
  - IGM band roles come from `band names` (lat / lon); without names, a band
    with |value| > 90 is longitude, else the band varying most down the rows
    (along track for a sun-synchronous orbit) is latitude.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.georef.from_scene_file import GeorefUnavailable

logger = logging.getLogger("GenericHsiHelper")

NO_MAP_MESSAGE = (
    "this file has no map coordinates (no CRS / map info in it, and no .igm geometry file beside it); "
    "export refuses rather than submit a map that cannot be placed"
)

# ENVI `data type` codes -> numpy dtype.
_ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}
_IGM_STEM_RE = re.compile(r"(^|[._-])igm$", re.IGNORECASE)


def is_igm(path: Path) -> bool:
    """`x.igm`, `x.igm.hdr`, `x_igm`, `x_igm.hdr`, `x_igm.img`."""
    p = Path(path)
    name = p.name.lower()
    if name.endswith(".igm") or name.endswith(".igm.hdr"):
        return True
    stem = p.stem if p.suffix else p.name
    return bool(_IGM_STEM_RE.search(stem))


def file_georef(src):
    """(Affine, CRS) GDAL reads from the file, or None when it has no CRS or only the identity transform."""
    if src.crs is None or src.transform.is_identity:
        return None
    return src.transform, src.crs


def find_igm(paths: Sequence[Path]) -> Optional[Tuple[Path, Path]]:
    """(header, data) of an IGM geometry file among `paths`, or None."""
    for hdr in sorted(p for p in paths if p.suffix.lower() == ".hdr" and is_igm(p)):
        direct = hdr.with_suffix("")
        if direct.is_file():
            return hdr, direct
        for p in sorted(paths):
            if p != hdr and p.parent == hdr.parent and p.stem == hdr.stem and p.suffix.lower() != ".hdr":
                return hdr, p
    if any(p.suffix.lower() == ".igm" for p in paths):
        logger.warning("an .igm file was uploaded without its own header (scene.igm.hdr); it cannot be read")
    return None


def read_igm(hdr: Path, data: Path) -> Tuple[np.ndarray, List[str], dict]:
    """(bands (B, H, W) float64, band names, header tokens). NaN where the header's ignore value sits."""
    from app.utils.files.envi_helper import _parse_envi_header

    tok = _parse_envi_header(hdr.read_text(errors="ignore"))
    lines, samples, bands = int(tok["lines"]), int(tok["samples"]), int(tok["bands"])
    dtype = np.dtype(_ENVI_DTYPES[int(tok.get("data type", "4"))])
    dtype = dtype.newbyteorder(">" if tok.get("byte order", "0").strip() == "1" else "<")
    offset = int(tok.get("header offset", "0"))
    raw = np.fromfile(data, dtype=dtype, count=lines * samples * bands, offset=offset).astype(np.float64)
    if raw.size != lines * samples * bands:
        raise GeorefUnavailable(f"IGM {data.name} is truncated: {raw.size} values for {bands}x{lines}x{samples}")
    inter = tok.get("interleave", "bsq").strip().lower()
    if inter == "bil":
        arr = raw.reshape(lines, bands, samples).transpose(1, 0, 2)
    elif inter == "bip":
        arr = raw.reshape(lines, samples, bands).transpose(2, 0, 1)
    else:
        arr = raw.reshape(bands, lines, samples)
    if "data ignore value" in tok:
        arr[arr == float(tok["data ignore value"])] = np.nan
    names = [n.strip().lower() for n in tok.get("band names", "").split(",") if n.strip()]
    return arr, names, tok


def igm_lat_lon(arr: np.ndarray, names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """(lat, lon), each (H, W). Raises GeorefUnavailable for projected (non-degree) coordinates."""
    if arr.shape[0] < 2:
        raise GeorefUnavailable("IGM has fewer than two bands")
    lat_i = next((i for i, n in enumerate(names) if "lat" in n), None)
    lon_i = next((i for i, n in enumerate(names) if "lon" in n), None)
    if lat_i is None or lon_i is None:
        a, b = arr[0], arr[1]
        if np.nanmax(np.abs(a)) > 90 >= np.nanmax(np.abs(b)):
            lat_i, lon_i = 1, 0
        elif np.nanmax(np.abs(b)) > 90 >= np.nanmax(np.abs(a)):
            lat_i, lon_i = 0, 1
        else:
            row_var = lambda x: float(np.nanmean(np.abs(np.diff(x, axis=0)))) if x.shape[0] > 1 else 0.0
            col_var = lambda x: float(np.nanmean(np.abs(np.diff(x, axis=1)))) if x.shape[1] > 1 else 0.0
            a_rowish = row_var(a) - col_var(a) >= row_var(b) - col_var(b)
            lat_i, lon_i = (0, 1) if a_rowish else (1, 0)
        logger.info("IGM band roles inferred: latitude = band %d, longitude = band %d", lat_i + 1, lon_i + 1)
    lat, lon = arr[lat_i], arr[lon_i]
    if np.nanmax(np.abs(lat)) > 90 or np.nanmax(np.abs(lon)) > 360:
        raise GeorefUnavailable("IGM holds projected coordinates, not latitude / longitude; only degrees are supported")
    return lat, lon


def igm_georef(lat: np.ndarray, lon: np.ndarray, target_shape):
    """EPSG:4326 affine fitted to the swath's lat/lon bounding box, as PRISMA's fallback does."""
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds

    ok = np.isfinite(lat) & np.isfinite(lon)
    if not ok.any():
        raise GeorefUnavailable("IGM has no finite latitude / longitude")
    h, w = target_shape
    transform = from_bounds(float(lon[ok].min()), float(lat[ok].min()), float(lon[ok].max()),
                            float(lat[ok].max()), w, h)
    return transform, CRS.from_epsg(4326)


def scene_georef(data_path: str, upload_paths: Sequence[Path], target_shape, cube_shape=None):
    """(Affine, CRS, source) with source "file" or "igm"; GeorefUnavailable(NO_MAP_MESSAGE) when neither exists.

    `cube_shape` is the file's (H, W) (defaults to target_shape): the IGM must
    match it, since an unrectified product has one geometry pixel per cube pixel.
    """
    from app.utils.files.generic_hsi_helper import open_raster

    with open_raster(data_path) as src:
        geo = file_georef(src)
        shape = (src.height, src.width)
    if geo is not None:
        if tuple(target_shape) != shape:
            raise GeorefUnavailable(f"raster {shape} in {Path(data_path).name} != target shape {tuple(target_shape)}")
        return geo[0], geo[1], "file"
    igm = find_igm(upload_paths)
    if igm is not None:
        arr, names, _ = read_igm(*igm)
        if arr.shape[1:] != tuple(cube_shape or target_shape):
            raise GeorefUnavailable(f"IGM {igm[1].name} is {arr.shape[1:]}, the cube is {tuple(cube_shape or target_shape)}")
        lat, lon = igm_lat_lon(arr, names)
        transform, crs = igm_georef(lat, lon, target_shape)
        return transform, crs, "igm"
    raise GeorefUnavailable(NO_MAP_MESSAGE)


def resolve_generic_hsi_georef(scene_dir: Path, target_shape):
    """The export path's entry point: (Affine, CRS) for `scene_dir`, or GeorefUnavailable."""
    from app.utils.files.generic_hsi_helper import locate_data

    paths = [p for p in Path(scene_dir).rglob("*") if p.is_file()]
    try:
        source, data_path, _ = locate_data(paths)
    except ValueError as exc:
        raise GeorefUnavailable(str(exc)) from exc
    transform, crs, how = scene_georef(data_path, paths, target_shape)
    logger.info("generic_hsi georef resolved from %s (%s): %s", Path(source).name, how, crs.to_string())
    return transform, crs
