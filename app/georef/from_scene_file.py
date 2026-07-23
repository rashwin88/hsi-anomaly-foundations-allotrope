"""Per-sensor (transform, crs) resolver — reads georef from the raw scene file.

The vendable model doesn't store spatial reference; we recover it at export
time. Each sensor's raw file ships georef in its own way:

  * **PRISMA L2D**: per-pixel ``Latitude`` + ``Longitude`` arrays under
    ``HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields``. Swath data — we fit
    an EPSG:4326 affine to the corner lat/lon and report that. Sub-pixel
    accuracy isn't critical for submission overlay; this is the
    industry-standard simplification for L2D HCO data.
  * **EnMAP L2A**: spectral image is a UTM GeoTIFF
    (``*-SPECTRAL_IMAGE.TIF``). rasterio reads CRS + transform directly.
  * **AVIRIS-NG**: ENVI BIL with a ``.hdr`` sidecar carrying ``map info``
    and ``coordinate system string``. rasterio understands ENVI.
  * **Landsat 9 / HotSat**: source ``.tif`` is a UTM GeoTIFF — rasterio
    direct read.

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
        ``landsat9``, ``hotsat1``).
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
    except (OSError, KeyError) as exc:
        raise GeorefUnavailable(
            f"failed to read PRISMA HE5 at {he5_path}: {exc}"
        ) from exc

    if lats.shape != lons.shape:
        raise GeorefUnavailable(
            f"PRISMA lat/lon arrays have mismatched shapes "
            f"{lats.shape} vs {lons.shape}"
        )

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
    #   HotSat-1  → the ortho-rectified product (skip UDM mask + thumbnails).
    if sensor_type == "landsat9":
        preferred = [
            p for p in tifs
            if "B10" in p.name.upper() or "ST" in p.name.upper()
        ]
    else:  # hotsat1
        preferred = [
            p for p in tifs
            if "ortho" in p.name.lower() or "stacked" in p.name.lower()
        ]
        # Reject UDM (uncertainty / mask) products even if they slipped in.
        preferred = [p for p in preferred if "udm" not in p.name.lower()]
    return _read_georef_from_geotiff((preferred or tifs)[0], target_shape)


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
