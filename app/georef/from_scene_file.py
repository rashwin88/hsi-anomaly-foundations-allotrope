"""Per-sensor (transform, crs) resolver — reads georef from the raw scene file.

The vendable model doesn't store spatial reference; we recover it at export
time. Each sensor's raw file ships georef in its own way:

  * **PRISMA L2D**: the product's own UTM grid from its root attributes
    (``Epsg_Code`` + ``Product_*corner_easting/northing``, ``prisma_utm_grid``) -
    the grid the evaluators' ground truth uses. Fallback when they are missing or
    the shape differs: an EPSG:4326 affine fitted to the per-pixel ``Latitude`` /
    ``Longitude`` bounding box (approximate, can be pixels off).
  * **EnMAP L2A**: spectral image is a UTM GeoTIFF
    (``*-SPECTRAL_IMAGE.TIF``). rasterio reads CRS + transform directly.
  * **AVIRIS-NG**: ENVI BIL with a ``.hdr`` sidecar carrying ``map info``
    and ``coordinate system string``. rasterio understands ENVI.
  * **Landsat 9 / HotSat**: source ``.tif`` is a UTM GeoTIFF — rasterio
    direct read.
  * **Tanager**: HDF-EOS5 GRID. The UTM grid comes from ``StructMetadata.0``
    (outer corners, so no half-pixel shift) and the grid group's
    ``epsg_code``, read with h5py by ``app.utils.files.tanager_helper``.
    An ENVI Tanager upload (``.hdr`` + data) is read by rasterio from the data
    file's ``map info`` / ``coordinate system string``; no map info refuses.
  * **generic_hsi**: the file's own CRS (any GDAL format, EPSG:4326 included),
    else an OSK ``.igm``'s per-pixel lat/lon fitted like PRISMA's fallback,
    else refuse with "this file has no map coordinates"
    (``app.utils.files.generic_hsi_georef``).

If the spatial extent (H, W) of the resolved georef doesn't match the
caller's expected shape, we error out instead of silently emitting a
mis-georeferenced bundle.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger("allotrope.app.georef")


class GeorefUnavailable(Exception):
    """No CRS could be recovered from the raw scene file on disk."""


# The export fallback for a scene with no georeference (user, 2026-11-01: always export). One unit
# per pixel, y negated so the grid reads north-up in GIS tools; written with crs=None, so no .prj.
# Built lazily: this module must import without rasterio.
def pixel_grid():
    """The pixel-coordinate affine: pixel centre (row, col) -> x = col + 0.5, y = -(row + 0.5)."""
    from rasterio.transform import Affine

    return Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)


NO_GEOREF_NOTE = ("no georeference: pixel coordinates (col,row); x=col+0.5, y=-(row+0.5) "
                  "at pixel centres")


def resolve_scene_georef(
    *,
    scene_dir: Path,
    sensor_type: str,
    target_shape: Tuple[int, int],
):
    """Return ``(transform, crs)`` for the scene.

    Parameters
    ----------
    scene_dir : Path
        ``<data_dir>/scenes/<scene_id>/raw`` — the per-scene raw folder.
    sensor_type : str
        Scene's ``sensor_type`` column (``prisma``, ``enmap``, ``aviris_ng``,
        ``landsat9``, ``hotsat1``, ``tanager``).
    target_shape : (H, W)
        Expected raster shape — the cube's spatial dims. Used to scale the
        derived affine if the source is at a different resolution (PRISMA),
        or to assert equality (everything else).

    Returns
    -------
    (rasterio.transform.Affine, rasterio.crs.CRS)

    Raises
    ------
    GeorefUnavailable
    """
    sensor_type = sensor_type.lower()
    if sensor_type == "prisma":
        return _resolve_prisma(scene_dir, target_shape)
    if sensor_type == "enmap":
        return _resolve_enmap(scene_dir, target_shape)
    if sensor_type == "aviris_ng":
        return _resolve_aviris_ng(scene_dir, target_shape)
    if sensor_type in ("landsat9", "hotsat1"):
        return _resolve_thermal_geotiff(scene_dir, sensor_type, target_shape)
    if sensor_type == "tanager":
        return _resolve_tanager(scene_dir, target_shape)
    if sensor_type == "generic_hsi":
        from app.utils.files.generic_hsi_georef import resolve_generic_hsi_georef

        return resolve_generic_hsi_georef(scene_dir, target_shape)
    raise GeorefUnavailable(f"unknown sensor_type {sensor_type!r}")


# ---------------------------------------------------------------------------
# PRISMA — swath data with per-pixel lat/lon arrays
# ---------------------------------------------------------------------------

_PRISMA_GEO_PATH = "HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields"
_PRISMA_GEO_PATH_FALLBACK = "HDFEOS/SWATHS/PRS_L2D_STD/Geolocation Fields"


def _resolve_prisma(scene_dir: Path, target_shape):
    import h5py
    import numpy as np
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    h5_files = (
        list(scene_dir.glob("*.he5")) + list(scene_dir.glob("*.HE5"))
        + list(scene_dir.rglob("*.he5")) + list(scene_dir.rglob("*.HE5"))
    )
    # Dedupe while preserving order (top-level matches preferred).
    seen: set = set()
    h5_files = [p for p in h5_files if not (p in seen or seen.add(p))]
    if not h5_files:
        raise GeorefUnavailable(
            f"no HE5 file under {scene_dir} — cannot resolve PRISMA georef"
        )
    he5_path = h5_files[0]

    try:
        with h5py.File(he5_path, "r") as f:
            geo = None
            for candidate in (_PRISMA_GEO_PATH, _PRISMA_GEO_PATH_FALLBACK):
                try:
                    geo = f[candidate]
                    break
                except KeyError:
                    continue
            if geo is None:
                raise GeorefUnavailable(
                    f"no Geolocation Fields group in {he5_path}"
                )
            lats = geo["Latitude"][:]
            lons = geo["Longitude"][:]
            attrs = dict(f.attrs)
    except (OSError, KeyError) as exc:
        raise GeorefUnavailable(
            f"failed to read PRISMA HE5 at {he5_path}: {exc}"
        ) from exc

    if lats.shape != lons.shape:
        raise GeorefUnavailable(
            f"PRISMA lat/lon arrays have mismatched shapes "
            f"{lats.shape} vs {lons.shape}"
        )

    grid = prisma_utm_grid(attrs, lats.shape)
    if grid is not None and tuple(target_shape) == tuple(lats.shape):
        logger.info("PRISMA georef resolved from %s: the product's own UTM grid, EPSG:%d", he5_path.name, grid[1])
        return grid[0], CRS.from_epsg(grid[1])

    # The source lat/lon arrays match the cube's source spatial shape, but
    # the vendable may have transposed / cropped. We fit the affine to the
    # corners regardless of source shape, then let rasterio fit it to the
    # caller's target_shape via from_bounds — which is sub-pixel-accurate
    # for axis-aligned PRISMA L2D scenes.
    min_lat, max_lat = float(np.nanmin(lats)), float(np.nanmax(lats))
    min_lon, max_lon = float(np.nanmin(lons)), float(np.nanmax(lons))

    H, W = target_shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)
    crs = CRS.from_epsg(4326)
    logger.info(
        "PRISMA georef resolved from %s: bbox lon[%.4f..%.4f] lat[%.4f..%.4f] → "
        "EPSG:4326 affine for %dx%d",
        he5_path.name, min_lon, max_lon, min_lat, max_lat, H, W,
    )
    return transform, crs


def prisma_utm_grid(attrs: dict, shape):
    """(Affine, EPSG) of a PRISMA L2D product's own map grid from its root attributes, or None if they are missing.

    L2D is gridded in UTM (`Epsg_Code`); the `Product_{UL,UR,LL}corner_{easting,northing}` attributes are the
    corner pixels' *centres*, so the pitch is the corner span over (n - 1) and the origin is the UL centre minus half
    a pixel. This is the grid the AIGC ground-truth rasters use (Set-1: EPSG:32642, [30, 0, 545007.75, 0, -30,
    2501437.5]); the lat/lon bounding-box fit below is only the fallback.
    """
    from rasterio.transform import Affine

    try:
        epsg = int(attrs["Epsg_Code"])
        ule, uln = float(attrs["Product_ULcorner_easting"]), float(attrs["Product_ULcorner_northing"])
        ure, lln = float(attrs["Product_URcorner_easting"]), float(attrs["Product_LLcorner_northing"])
    except (KeyError, TypeError, ValueError):
        return None
    h, w = shape
    if epsg <= 0 or h < 2 or w < 2 or ure <= ule or uln <= lln:
        return None
    dx, dy = round((ure - ule) / (w - 1), 2), round((uln - lln) / (h - 1), 2)   # float32 attrs: 29.9998 -> 30.0
    return Affine(dx, 0.0, ule - dx / 2, 0.0, -dy, uln + dy / 2), epsg


# ---------------------------------------------------------------------------
# EnMAP — UTM GeoTIFF, ride rasterio
# ---------------------------------------------------------------------------

def _resolve_enmap(scene_dir: Path, target_shape):
    # EnMAP raw layout nests the scene folder one level inside `raw/`:
    #   raw/ENMAP01-...-V010506.../ENMAP01-...-V010506...-SPECTRAL_IMAGE.TIF
    # Walk recursively to find it; matches both rglob casings.
    spectral = (
        list(scene_dir.rglob("*SPECTRAL_IMAGE.TIF"))
        + list(scene_dir.rglob("*SPECTRAL_IMAGE.tif"))
    )
    if not spectral:
        raise GeorefUnavailable(
            f"no *SPECTRAL_IMAGE.TIF found anywhere under {scene_dir}"
        )
    return _read_georef_from_geotiff(spectral[0], target_shape)


# ---------------------------------------------------------------------------
# AVIRIS-NG — ENVI BIL, ride rasterio
# ---------------------------------------------------------------------------

def _resolve_aviris_ng(scene_dir: Path, target_shape):
    # AVIRIS-NG products ship as ENVI BIL: a binary file + sidecar .hdr.
    # Layout is typically nested one folder deep under `raw/`:
    #   raw/ang<TS>/ang<TS>_corr_v2m2_img.bin   ← reflectance (georef-bearing)
    #   raw/ang<TS>/ang<TS>_corr_v2m2_img.hdr   ← ENVI header
    #   raw/ang<TS>/ang<TS>_h2o_v2m2_img.bin    ← aux water vapor
    #   raw/ang<TS>/ang<TS>_h2o_v2m2_img.hdr
    # We want the reflectance product (`_corr_`), not the h2o one.
    # rasterio opens the .hdr directly via the ENVI driver, which is the
    # most robust route — it handles the rotation parameter on `map info`.
    hdrs = list(scene_dir.glob("*.hdr")) or list(scene_dir.rglob("*.hdr"))
    # Skip checksum files (.sha256) and any non-product .hdr's.
    hdrs = [p for p in hdrs if p.suffix.lower() == ".hdr"]
    if not hdrs:
        raise GeorefUnavailable(f"no ENVI .hdr under {scene_dir}")
    # Prefer the reflectance / orthorectified product (`_corr_` or `_rfl_`).
    def _score(p):
        n = p.stem.lower()
        return (
            "corr" not in n,   # reflectance product wins
            "rfl" not in n,    # legacy AVIRIS-Classic naming
            "h2o" in n,         # h2o is the explicit deprioritisation
            p.name,
        )
    hdrs.sort(key=_score)

    # Newer rasterio refuses to open the .hdr directly — it requires the
    # binary file path. Find the binary that lives next to the chosen
    # header with the same stem. AVIRIS-NG ships them as `.bin`, but
    # older campaigns sometimes use `.img` or no extension at all.
    hdr = hdrs[0]
    stem = hdr.stem
    folder = hdr.parent
    binary: Path | None = None
    for ext in (".bin", ".img", ".dat", ""):
        cand = folder / (stem + ext) if ext else folder / stem
        # Skip checksum files; pick the largest matching candidate (the cube).
        if (
            cand.is_file()
            and cand.suffix.lower() not in (".sha256", ".md5", ".hdr")
            and cand != hdr
        ):
            binary = cand
            break
    if binary is None:
        # Last-resort: any sibling with the same stem that isn't a checksum
        # or a header.
        for cand in folder.iterdir():
            if (
                cand.is_file()
                and cand.stem == stem
                and cand.suffix.lower() not in (".sha256", ".md5", ".hdr")
            ):
                binary = cand
                break
    if binary is None:
        raise GeorefUnavailable(
            f"found AVIRIS-NG header {hdr.name} but no matching binary"
        )
    return _read_georef_from_geotiff(binary, target_shape)


# ---------------------------------------------------------------------------
# Landsat 9 + HotSat — already a GeoTIFF
# ---------------------------------------------------------------------------

def _resolve_thermal_geotiff(scene_dir: Path, sensor_type: str, target_shape):
    # Thermal sensors ship a georeferenced TIFF. Naming varies:
    #   Landsat 9  → *_ST_B10.TIF or *_B10.TIF       (top of `raw/`)
    #   HotSat-1   → *_ortho.tiff / *_stacked.tiff   (one level deeper)
    # Top-level first, fall back to recursive walk; both `.tif` + `.tiff`.
    patterns = ("*.tif", "*.TIF", "*.tiff", "*.TIFF")
    tifs: list[Path] = []
    for pat in patterns:
        tifs.extend(scene_dir.glob(pat))
    if not tifs:
        for pat in patterns:
            tifs.extend(scene_dir.rglob(pat))
    if not tifs:
        raise GeorefUnavailable(
            f"no .tif/.tiff under {scene_dir} for {sensor_type}"
        )
    # Pick the most georef-likely file per sensor:
    #   Landsat 9 → the thermal band (B10 / ST).
    #   HotSat-1  → the raster the cube was built from (never the UDM mask).
    if sensor_type == "hotsat1":
        return _read_georef_from_geotiff(_hotsat_cube_tiff(scene_dir, tifs), target_shape)
    preferred = [
        p for p in tifs
        if "B10" in p.name.upper() or "ST" in p.name.upper()
    ]
    return _read_georef_from_geotiff((preferred or tifs)[0], target_shape)


def _hotsat_cube_tiff(scene_dir: Path, tifs: list[Path]) -> Path:
    """The GeoTIFF the HotSat builder read the cube from, so the export grid is the cube's grid.

    The ortho and the stacked rasters sit on different grids, so glob order must not choose.
    HotSatHelper's ``visual`` asset is the builder's own choice; without a readable
    metadata.json, the ortho then the stacked raster, by sorted name. Never the UDM.
    """
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.files.hotsat_helper import HotSatHelper

    for meta in sorted(scene_dir.rglob("metadata.json")):
        try:
            return Path(HotSatHelper(FileSourceConfig(source_path=str(meta.parent))).primary_visual_path)
        except (OSError, ValueError) as exc:
            logger.warning("HotSat metadata %s unusable for georef (%s); picking by file name", meta, exc)
    named = sorted((p for p in tifs if "udm" not in p.name.lower()), key=lambda p: p.name)
    for suffix in ("_ortho.tiff", "_stacked.tiff"):
        for p in named:
            if p.name.lower().endswith(suffix):
                return p
    raise GeorefUnavailable(f"no HotSat *_ortho.tiff / *_stacked.tiff under {scene_dir}")


# ---------------------------------------------------------------------------
# Tanager — HDF-EOS5 grid geometry
# ---------------------------------------------------------------------------

def _resolve_tanager(scene_dir: Path, target_shape):
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    from app.utils.files.tanager_helper import grid_epsg, pick_tanager_file, read_tanager_metadata

    path = pick_tanager_file([p for p in scene_dir.rglob("*") if p.is_file()])
    if path is None:
        raise GeorefUnavailable(f"no Tanager .h5, or ENVI .hdr with its data file, under {scene_dir}")
    if path.suffix.lower() == ".hdr":
        return _resolve_tanager_envi(path, target_shape)
    try:
        grid = read_tanager_metadata(path).grid
        epsg = grid_epsg(grid)
    except (OSError, KeyError, ValueError) as exc:
        raise GeorefUnavailable(f"failed to read the Tanager grid from {path}: {exc}") from exc
    # No rescaling, unlike _read_georef_from_geotiff: band_filter_apply never
    # crops a Tanager cube, so a shape mismatch means the wrong file.
    if tuple(target_shape) != (grid.y_dim, grid.x_dim):
        raise GeorefUnavailable(
            f"Tanager grid {grid.y_dim}x{grid.x_dim} in {path.name} != target shape {tuple(target_shape)}"
        )
    dx, dy = grid.pixel_size_m
    ulx, uly = grid.upper_left_m
    logger.info("Tanager georef resolved from %s: EPSG:%d, %.2f m grid", path.name, epsg, dx)
    return Affine(dx, 0.0, ulx, 0.0, -dy, uly), CRS.from_epsg(epsg)


def _resolve_tanager_envi(hdr: Path, target_shape):
    """CRS + affine GDAL reads from the header's map info. Exact shape, as for the HDF5 grid."""
    import warnings

    import rasterio

    from app.utils.files.tanager_envi import envi_data_file

    data = envi_data_file(hdr)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
            with rasterio.open(data) as src:
                crs, transform, shape = src.crs, src.transform, (src.height, src.width)
    except Exception as exc:
        raise GeorefUnavailable(f"rasterio could not open {data}: {exc}") from exc
    if crs is None:
        raise GeorefUnavailable(
            f"ENVI header {hdr.name} has no map info / coordinate system string: the scene cannot be georeferenced"
        )
    if shape != tuple(target_shape):
        raise GeorefUnavailable(f"Tanager ENVI raster {shape} in {data.name} != target shape {tuple(target_shape)}")
    logger.info("Tanager georef resolved from ENVI %s: %s", hdr.name, crs.to_string())
    return transform, crs


# ---------------------------------------------------------------------------
# Shared geotiff georef reader (used by everything except PRISMA)
# ---------------------------------------------------------------------------

def _read_georef_from_geotiff(path: Path, target_shape):
    import rasterio
    from rasterio.transform import Affine

    try:
        with rasterio.open(path) as src:
            crs = src.crs
            transform = src.transform
            src_shape = src.height, src.width
    except Exception as exc:
        raise GeorefUnavailable(
            f"rasterio could not open {path}: {exc}"
        ) from exc

    if crs is None:
        raise GeorefUnavailable(f"{path} has no CRS")

    if src_shape == tuple(target_shape):
        return transform, crs

    # Vendable may have been cropped/transposed. Scale the affine to map
    # the target_shape onto the source's spatial extent, preserving the
    # outer bounding box.
    H_src, W_src = src_shape
    H_tgt, W_tgt = target_shape
    sx = W_src / W_tgt
    sy = H_src / H_tgt
    # The source affine: (a, b, c, d, e, f) means
    #   X = a*col + b*row + c
    #   Y = d*col + e*row + f
    # Scaling cols by sx, rows by sy keeps the corners pinned.
    scaled = Affine(
        transform.a * sx, transform.b * sy, transform.c,
        transform.d * sx, transform.e * sy, transform.f,
    )
    logger.info(
        "georef scaled from source shape %s to target %s (sx=%.4f sy=%.4f)",
        src_shape, target_shape, sx, sy,
    )
    return scaled, crs
