"""
One EnMAP L2A scene folder, read for training shards: its native bands and wavelengths, the
provider quality layers, and the scene facts a cloud/water head needs to decide when to trust them.
`band_index` maps SpectralEarth's 202 wavelengths onto the scene's bands for load-time selection.

Used by app/utils/enmap/tiles.py, which cuts the scene into patches, and through it by
scripts/shard_enmap.py. The product path (EnmapDatasetBuilder) is separate: it resamples onto
the 165-band grid and applies the quality masks; this keeps native bands and masks untouched.

Why bands are chosen by wavelength, never by band number: the band-number-to-wavelength map
is not fixed. 5 of the 210 scenes imported on 2026-10-14 (early-2023 acquisitions, processor
V010502) number their SWIR one band off from every other scene - band 142 is 1530.29 nm
there and 1518.87 nm elsewhere - so selecting SpectralEarth's "EnMAP bands 1-126, 142-160,
168-224" by number would mislabel 19 bands by ~11 nm. Every scene does contain all 202
SpectralEarth wavelengths exactly; matching on them is right for both maps.

Gotchas:
  - Files are found by suffix (`-SPECTRAL_IMAGE_COG.tiff`, `-SPECTRAL_IMAGE.TIF`, ...): the
    inventory keeps provider names, and older deliveries are not COGs.
  - Quality-layer values are kept exactly as delivered (cirrus 0-3; classes 0 cloud,
    1 land, 2 water, 3 no-data) - see docs/03-data-pipeline.md, "Segmentation shards".
  - spatial_info width/height in METADATA.XML are not the raster size; read the raster.
"""

import os
import re
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from app.models.dataset.spectralearth import load_band_table
from app.utils.files.enmap_xml_parser import EnmapXmlParser

# Exact match expected (all 210 scenes differ by 0.0 nm from SpectralEarth's table); the
# tolerance only absorbs rounding in the XML.
WAVELENGTH_TOLERANCE_NM = 0.5

# label key -> file suffix stem; keys match the enmap_seg lane so one reader serves both.
LABEL_LAYERS = {
    "label_cloud": "QL_QUALITY_CLOUD",
    "label_cirrus": "QL_QUALITY_CIRRUS",
    "label_haze": "QL_QUALITY_HAZE",
    "label_cloud_shadow": "QL_QUALITY_CLOUDSHADOW",
    "label_snow": "QL_QUALITY_SNOW",
    "label_classes": "QL_QUALITY_CLASSES",
}


class SceneError(ValueError):
    """The folder is not a usable EnMAP L2A scene (missing file, missing wavelength)."""


def find_member(folder: str, stem: str) -> str:
    """Path of the scene file whose name ends in `-<stem>` + a raster/XML extension. Case-
    insensitive: deliveries mix `-METADATA.XML` and `-METADATA.xml`, and a case-sensitive lookup
    silently skipped those scenes on Linux in the October H200 trial."""
    pattern = re.compile(rf"-{stem}(_COG)?\.(tiff?|xml)$", re.IGNORECASE)
    hits = [os.path.join(folder, f) for f in os.listdir(folder) if pattern.search(f)]
    if len(hits) != 1:
        raise SceneError(f"{folder}: expected one *-{stem} file, found {len(hits)}")
    return hits[0]


def select_bands(scene_wavelengths_nm: np.ndarray, wanted_nm: np.ndarray,
                 tolerance_nm: float = WAVELENGTH_TOLERANCE_NM) -> np.ndarray:
    """0-based scene band index for each wanted wavelength, in the wanted order.
    Raises SceneError if any wanted wavelength has no band within tolerance."""
    dist = np.abs(np.asarray(scene_wavelengths_nm)[None, :] - np.asarray(wanted_nm)[:, None])
    idx = dist.argmin(axis=1)
    worst = dist[np.arange(len(idx)), idx].max()
    if worst > tolerance_nm:
        raise SceneError(f"a wanted wavelength is {worst:.2f} nm from the nearest scene band")
    return idx


def _xml_mean(xml: str, tag: str) -> float | None:
    """Mean of the numbers inside <tag>...</tag> (a scalar, or EnMAP's four-corner blocks)."""
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S)
    if not m:
        return None
    nums = [float(v) for v in re.findall(r">\s*(-?[\d.]+)\s*<", f">{m.group(1)}<")]
    return float(np.mean(nums)) if nums else None


@dataclass
class EnmapScene:
    folder: str
    scene_id: str = field(init=False)

    def __post_init__(self):
        self.scene_id = os.path.basename(os.path.normpath(self.folder))

    @cached_property
    def _xml_path(self) -> str:
        return find_member(self.folder, "METADATA")

    @cached_property
    def metadata(self):
        return EnmapXmlParser(self._xml_path).parse()

    @cached_property
    def wavelengths_nm(self) -> np.ndarray:
        """(224,) centre wavelengths in band-number order."""
        bands = sorted(self.metadata.band_characterisation, key=lambda b: b.band_number)
        return np.array([b.wavelength_center for b in bands])

    @cached_property
    def band_index(self) -> np.ndarray:
        """(202,) 0-based scene band for each SpectralEarth band, in SpectralEarth's order."""
        return select_bands(self.wavelengths_nm, load_band_table().wavelengths_nm)

    @cached_property
    def facts(self) -> dict:
        """Scene-level facts carried into every patch's meta.json."""
        xml = open(self._xml_path, encoding="utf-8", errors="replace").read()
        md, q = self.metadata, self.metadata.quality_flags
        return {
            "scene_id": self.scene_id, "datatake": f"DT{md.datatake_id}", "tile": md.tile_id,
            "acquired_at": md.start_time.isoformat() + "Z",
            "processor": (re.search(r"_V(\d{6})_", self.scene_id) or [None, None])[1],
            "mean_ground_elevation_m": _xml_mean(xml, "meanGroundElevation"),
            "scene_wv": _xml_mean(xml, "sceneWV"), "scene_aot": _xml_mean(xml, "sceneAOT"),
            "sun_elevation_deg": _xml_mean(xml, "sunElevationAngle"),
            "sun_azimuth_deg": _xml_mean(xml, "sunAzimuthAngle"),
            "cover_pct": {"cloud": q.cloud_cover, "cirrus": q.cirrus_cover, "haze": q.haze_cover,
                          "snow": q.snow_cover, "water": q.water_cover, "cloud_shadow": q.cloud_shadow},
        }
