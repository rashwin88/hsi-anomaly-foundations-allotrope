"""
Writers for the synthetic "surprise" files the generic_hsi tests read: GeoTIFF / COG, ENVI, HDF5, IGM.

Shared by tests/test_utils/test_generic_hsi.py and
tests/test_backend/test_generic_hsi_pipeline.py. Everything is written into
pytest's tmp_path at a few pixels, so nothing large is committed.

The cube encodes its own wavelengths: band b holds wl[b] / 10000 plus a small
per-pixel ramp, so a test can check from the values alone that bands were
reordered together with their wavelengths.
"""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import rasterio
from rasterio.transform import from_origin

H, W = 12, 16
WL = np.linspace(420.0, 2480.0, 40)
UTM_TRANSFORM = from_origin(589050.0, 3653460.0, 30.0, 30.0)
GEO_TRANSFORM = from_origin(-104.1, 32.9, 0.0003, 0.0003)


def cube_for(wl: Sequence[float], scale: float = 1.0, dtype="float32") -> np.ndarray:
    """(C, H, W): band b = wl[b] / 10000 + a 0..0.01 ramp, times `scale`."""
    ramp = (np.arange(H)[:, None] * W + np.arange(W)[None, :]) / (H * W) * 0.01
    cube = np.stack([np.full((H, W), w / 10000.0) + ramp for w in wl]) * scale
    return cube.astype(dtype)


def write_gtiff(path: Path, cube: np.ndarray, *, band_tags: Optional[Sequence[Dict[str, str]]] = None,
                descriptions: Optional[Sequence[str]] = None, nodata=None, crs=None, transform=None,
                root_tags: Optional[Dict[str, str]] = None, cog: bool = False) -> Path:
    """A pixel-interleaved GeoTIFF; `cog` copies it through GDAL's COG driver."""
    c, h, w = cube.shape
    tmp = path.with_suffix(".plain.tif") if cog else path
    profile = dict(driver="GTiff", count=c, height=h, width=w, dtype=cube.dtype.name, interleave="pixel")
    if crs is not None:
        profile.update(crs=crs, transform=transform)
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(cube)
        for i in range(c):
            if band_tags is not None:
                dst.update_tags(i + 1, **band_tags[i])
            if descriptions is not None:
                dst.set_band_description(i + 1, descriptions[i])
        if root_tags:
            dst.update_tags(**root_tags)
    if cog:
        from rasterio.shutil import copy

        copy(tmp, path, driver="COG")
        tmp.unlink()
    return path


def wl_tags(wl: Sequence[float], units: str = "nm", key: str = "wavelenght_units") -> list:
    """GHOSt's per-band metadata, misspelt units key and all."""
    return [{"wavelength": f"{w:.9g}", key: units} for w in wl]


def write_envi(folder: Path, stem: str, cube: np.ndarray, *, interleave: str = "bsq", ext: str = ".hsi",
               header: Optional[Dict[str, str]] = None, wl: Optional[Sequence[float]] = None,
               fwhm: Optional[Sequence[float]] = None, map_info: Optional[str] = None) -> Path:
    """ENVI raw data in BSQ / BIL / BIP plus its `.hdr`; returns the header path."""
    c, h, w = cube.shape
    dtype_code = {"float32": 4, "float64": 5, "uint16": 12, "int16": 2}[cube.dtype.name]
    layout = {"bsq": cube, "bil": cube.transpose(1, 0, 2), "bip": cube.transpose(1, 2, 0)}[interleave]
    folder.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(layout).astype(cube.dtype.newbyteorder("<")).tofile(folder / f"{stem}{ext}")
    lines = ["ENVI", f"samples = {w}", f"lines = {h}", f"bands = {c}", "header offset = 0", "file type = ENVI Standard",
             f"data type = {dtype_code}", f"interleave = {interleave}", "byte order = 0"]
    if wl is not None:
        lines.append("wavelength = {" + ", ".join(f"{v:.4f}" for v in wl) + "}")
    if fwhm is not None:
        lines.append("fwhm = {" + ", ".join(f"{v:.4f}" for v in fwhm) + "}")
    if map_info:
        lines.append("map info = {" + map_info + "}")
    for k, v in (header or {}).items():
        lines.append(f"{k} = {v}")
    hdr = folder / f"{stem}.hdr"
    hdr.write_text("\n".join(lines) + "\n")
    return hdr


def write_igm(folder: Path, stem: str, lat: np.ndarray, lon: np.ndarray, names=True) -> Path:
    """OSK-style geometry: `<stem>.igm` + `<stem>.igm.hdr`, BSQ float64 bands lat, lon, elevation."""
    arr = np.stack([lat, lon, np.full_like(lat, 1200.0)]).astype("<f8")
    arr.tofile(folder / f"{stem}.igm")
    h, w = lat.shape
    text = ["ENVI", f"samples = {w}", f"lines = {h}", "bands = 3", "header offset = 0", "data type = 5",
            "interleave = bsq", "byte order = 0"]
    if names:
        text.append("band names = {Latitude, Longitude, Elevation}")
    (folder / f"{stem}.igm.hdr").write_text("\n".join(text) + "\n")
    return folder / f"{stem}.igm"


def swath(h=H, w=W):
    """A slightly rotated north-south swath near 32.8 N, 104 W."""
    r, c = np.mgrid[0:h, 0:w].astype(np.float64)
    lat = 32.80 - r * 0.0003 + c * 0.00002
    lon = -104.00 + c * 0.0003 + r * 0.00002
    return lat, lon
