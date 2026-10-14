"""
One SpectralEarth patch, with the spectral meaning its file doesn't carry.

Built by app/utils/spectralearth/source.py from bytes in the inventory collection
SPECTRAL_EARTH_PATCHES; consumed by the screens, the torch dataset, and to_vendable()
(the bridge onto the 165-band common grid).

DLR ships each patch as a bare 202-band int16 GeoTIFF: no wavelengths, no georeference,
no date, no quality masks. The band table (app/constants/spectralearth_bands.json, made by
scripts/build_spectralearth_band_table.py) restores what the bands are; this module is the
one place that turns raw counts into reflectance and nodata into validity, so every
consumer agrees on both.

Gotchas:
  - Raw is EnMAP L2A reflectance x 10,000 as int16; nodata is -32768 per band per pixel.
  - Wavelength is not monotonic in band order: the VNIR and SWIR detectors overlap
    around 900-1000 nm. Sort by wavelength before interpolating; take family from the
    table, never from wavelength.
  - Shapes are (bands, rows, cols) = (202, 128, 128).
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from rasterio.io import MemoryFile

from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

BAND_TABLE_PATH = Path(__file__).resolve().parents[2] / "constants" / "spectralearth_bands.json"
PATCH_SIZE = 128


@dataclass(frozen=True)
class SpectralEarthBands:
    """The 202-band table: what each plane of a SpectralEarth patch means."""

    wavelengths_nm: np.ndarray          # (202,) float64, band order (not sorted)
    fwhm_nm: np.ndarray                 # (202,) float64
    families: tuple[SpectralFamily, ...]
    enmap_bands: tuple[int, ...]        # 1-based EnMAP band each plane came from
    reflectance_scale: float
    nodata: int

    @property
    def count(self) -> int:
        return len(self.families)


@cache
def load_band_table(path: Path = BAND_TABLE_PATH) -> SpectralEarthBands:
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    b = t["bands"]
    return SpectralEarthBands(
        wavelengths_nm=np.array([x["wavelength_nm"] for x in b], dtype=np.float64),
        fwhm_nm=np.array([x["fwhm_nm"] for x in b], dtype=np.float64),
        families=tuple(SpectralFamily(x["family"]) for x in b),
        enmap_bands=tuple(x["enmap_band"] for x in b),
        reflectance_scale=float(t["reflectance_scale"]),
        nodata=int(t["nodata"]),
    )


def parse_member_path(path: str) -> tuple[str, str]:
    """`enmap/0019846/01121.tif` -> ("0019846", "01121")."""
    parts = path.split("/")
    if len(parts) < 2 or not parts[-1].endswith(".tif"):
        raise ValueError(f"not a SpectralEarth patch path: {path}")
    return parts[-2], parts[-1][: -len(".tif")]


@dataclass
class SpectralEarthPatch:
    """A raw patch plus the band table that gives it meaning."""

    site_id: str
    patch_id: str
    raw: np.ndarray                     # (202, 128, 128) int16, exactly as shipped
    bands: SpectralEarthBands

    @classmethod
    def from_geotiff_bytes(
        cls, data: bytes, path: str, bands: SpectralEarthBands | None = None
    ) -> "SpectralEarthPatch":
        """Decode one patch file. Raises ValueError if it isn't 202 x 128 x 128 int16."""
        bands = bands or load_band_table()
        site_id, patch_id = parse_member_path(path)
        with MemoryFile(data) as mem, mem.open() as src:
            raw = src.read()
        if raw.shape != (bands.count, PATCH_SIZE, PATCH_SIZE) or raw.dtype != np.int16:
            raise ValueError(f"{path}: expected int16 {(bands.count, PATCH_SIZE, PATCH_SIZE)}, "
                             f"got {raw.dtype} {raw.shape}")
        return cls(site_id=site_id, patch_id=patch_id, raw=raw, bands=bands)

    @property
    def key(self) -> str:
        """Stable id, `se_<site>_<patch>`. Usable as a WebDataset key: no `.` (WebDataset
        splits the extension at the first dot) and no `/` (it would become a tar path)."""
        return f"se_{self.site_id}_{self.patch_id}"

    @property
    def wavelengths_nm(self) -> np.ndarray:
        return self.bands.wavelengths_nm

    @property
    def families(self) -> tuple[SpectralFamily, ...]:
        return self.bands.families

    def validity(self) -> np.ndarray:
        """(202, 128, 128) int8: 1 where the band has data, 0 where it is nodata."""
        return (self.raw != self.bands.nodata).astype(np.int8)

    def reflectance(self) -> np.ndarray:
        """(202, 128, 128) float32 surface reflectance; nodata voxels are 0.0, the same
        convention the EnMAP builder and the training shards use."""
        out = self.raw.astype(np.float32) * np.float32(self.bands.reflectance_scale)
        out[self.raw == self.bands.nodata] = 0.0
        return out
