"""
One AVIRIS-NG L2 flight line, read strip by strip at its native bands, for the training shards.

Used by app/utils/aviris/tiles.py and scripts/shard_aviris.py. The product reads AVIRIS-NG through
AvirisNgDatasetBuilder, whose file lookup expects the `_corr_v2m2` suffix; the India campaign ships
v2m2, v2m3, v2s1 and v2s2, so this finds the `_corr_*_img` reflectance pair by pattern and reuses
the builder's ENVI header parsing (app/utils/files/envi_helper.py).

    reflectance = the stored float32 (the `_corr_` cube is surface reflectance despite the header's
                  "Measured Radiances" description - median 0.08 at 550 nm, checked 2026-10-15)
    nodata      = data ignore value (-9999)
    dropped     = bands with bbl = 0 (53 of 425: the water-vapour windows); 372 kept

Gotchas:
  - The .bin is BIL, (lines, bands, samples), up to ~11 GB: it is memory-mapped and read one
    128-line strip at a time, never whole.
  - Pixel size varies by flight line (3.2-8.2 m seen); it travels in facts / meta.json.
  - Band centres differ by up to 0.42 nm between product versions - inside the 0.5 nm tolerance
    of app/utils/band_alignment.py, so lines can share batches once aligned.
"""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.utils.files.envi_helper import _build_core, _build_map_info, _build_spectral_axis, _parse_envi_header

NODATA = -32768
REFLECTANCE_SCALE = 0.0001
_DTYPES = {4: "f4", 5: "f8", 2: "i2", 12: "u2"}


@dataclass
class AvirisScene:
    scene_id: str               # e.g. ang20180318t060114
    bin_path: Path
    lines: int
    samples: int
    bands: int
    good: np.ndarray            # (bands,) bool, bbl == 1
    wavelengths_nm: np.ndarray  # (good,) float32
    ignore_value: float
    dtype: np.dtype
    offset: int
    facts: dict = field(default_factory=dict)

    def strip(self, y0: int, height: int) -> np.ndarray:
        """Lines y0..y0+height as int16 (good bands, height, samples), reflectance x 10,000."""
        cube = np.memmap(self.bin_path, dtype=self.dtype, mode="r", offset=self.offset, shape=(self.lines, self.bands, self.samples))
        raw = np.asarray(cube[y0:y0 + height][:, self.good, :]).transpose(1, 0, 2)    # (C, h, W)
        bad = (raw <= self.ignore_value + 0.5) | ~np.isfinite(raw)
        px = np.clip(np.rint(raw / REFLECTANCE_SCALE), -32767, 32767).astype(np.int16)
        px[bad] = NODATA
        return px

    def lonlat(self, col: float, row: float) -> tuple[float, float]:
        """(lon, lat) of a pixel centre, from the header's rotated UTM map info."""
        from rasterio.warp import transform
        m = self.facts["map"]
        x, y = (col + 0.5 - (m["ref_x"] - 1)) * m["px"], (row + 0.5 - (m["ref_y"] - 1)) * m["py"]
        t = math.radians(m["rotation"])
        e = m["easting"] + x * math.cos(t) + y * math.sin(t)
        n = m["northing"] + x * math.sin(t) - y * math.cos(t)
        epsg = (32600 if m["north"] else 32700) + m["zone"]
        (lon,), (lat,) = transform(f"EPSG:{epsg}", "EPSG:4326", [e], [n])
        return lon, lat

    @property
    def centre(self) -> tuple[float, float]:
        """(lat, lon) of the image centre."""
        lon, lat = self.lonlat(self.samples / 2, self.lines / 2)
        return lat, lon


def open_scene(folder: str | Path) -> AvirisScene:
    """The `_corr_*_img` reflectance pair in a flight-line folder. Raises FileNotFoundError if
    there is not exactly one, ValueError if the .bin size disagrees with its header."""
    folder = Path(folder)
    hdrs = sorted(folder.glob("*_corr_*_img.hdr"))
    if len(hdrs) != 1:
        raise FileNotFoundError(f"{folder}: expected one *_corr_*_img.hdr, found {len(hdrs)}")
    hdr, binf = hdrs[0], hdrs[0].with_suffix(".bin")
    tok = _parse_envi_header(hdr.read_text())
    core = _build_core(tok)
    sp = _build_spectral_axis(tok, core.bands)
    mi = _build_map_info(tok)
    if core.interleave != "bil":
        raise ValueError(f"{hdr.name}: interleave {core.interleave}, expected bil")
    dtype = np.dtype(("<" if core.byte_order == 0 else ">") + _DTYPES[core.data_type])
    if binf.stat().st_size != core.header_offset + core.lines * core.bands * core.samples * dtype.itemsize:
        raise ValueError(f"{binf.name}: size does not match its header")
    good = np.asarray(sp.bbl if sp.bbl else [1] * core.bands) == 1
    sid = re.match(r"ang\d{8}t\d{6}", hdr.name).group(0)
    facts = {"scene_id": sid, "product": hdr.name.split("_")[2], "date": sid[3:11],
             "pixel_size_m": float(mi.pixel_size_x) if mi else None,
             "map": {"ref_x": mi.reference_pixel_x, "ref_y": mi.reference_pixel_y, "easting": mi.reference_easting,
                     "northing": mi.reference_northing, "px": mi.pixel_size_x, "py": mi.pixel_size_y,
                     "zone": mi.utm_zone, "north": str(mi.utm_hemisphere).lower().startswith("n"),
                     "rotation": mi.rotation_degrees} if mi else None}
    return AvirisScene(sid, binf, core.lines, core.samples, core.bands, good,
                       np.asarray(sp.wavelengths, np.float32)[good],
                       -9999.0 if core.data_ignore_value is None else float(core.data_ignore_value), dtype,
                       core.header_offset, facts)
