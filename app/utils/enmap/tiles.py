"""
Cut an EnMAP scene into 128 x 128 training samples at its native bands, plus the quality layers
and facts a cloud/water segmentation head will need.

Called by scripts/shard_enmap.py (pass 1: scene -> per-tile files); the shards themselves are
written by app/utils/spectralearth/sharding.py's write_samples, so EnMAP and SpectralEarth
shards are read by the same trainer code.

Why the whole scene is read at once: the rasters are 512 x 512-block COGs, so a 128 x 128
window still decompresses its whole block - reading tile by tile in shuffled order would
decode every block up to 16 times. One scene is ~0.6 GB of int16 in memory.

Why native bands (decided 2026-10-14): the wavelength-token stems read any band list, so nothing
is chosen or resampled at shard time. Every one of the 224 bands is kept with the scene's own
wavelengths; the trainer picks at load time (e.g. SpectralEarth's 202 for the fixed-band stems,
or for batches mixed with SpectralEarth). 5 of 210 scenes number their bands differently - their
wavelengths.npy says so, per sample.

Sample keys (per tile):
  pixels.npy        int16 (224, 128, 128) - all EnMAP bands in band-number order; x 0.0001 =
                    reflectance; -32768 = nodata, including every voxel QL_PIXELMASK flags (the
                    water-vapour bands are largely blanked to nodata by L2A itself)
  wavelengths.npy   float32 (224,), this scene's band centres
  label_*.npy       uint8 (1, 128, 128), the six provider quality layers, never applied
  meta.json         scene facts (EnmapScene.facts) + tile row/col, pixel origin, centre
                    lon/lat, CRS and the tile's affine transform

Gotchas:
  - Partial edge tiles are dropped: 1192 x 1145 gives 9 x 8 full tiles; the strips beyond are
    not padded.
  - A tile is kept when at least min_valid_fraction of its pixels have any valid band; the
    trainer applies its own, stricter 0.4 on top.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

from app.utils.enmap.scene import LABEL_LAYERS, EnmapScene, find_member

TILE = 128
NODATA = -32768
# Swath edges: a tile under 10% valid is almost all off-swath padding and teaches nothing.
MIN_VALID_FRACTION = 0.10


@dataclass
class SceneArrays:
    """A scene read once: all bands with the pixel mask applied, and the label layers."""

    pixels: np.ndarray            # (224, H, W) int16, all bands
    labels: dict[str, np.ndarray] # key -> (H, W) uint8
    transform: rasterio.Affine
    crs: str


def read_scene(scene: EnmapScene) -> SceneArrays:
    with rasterio.open(find_member(scene.folder, "SPECTRAL_IMAGE")) as src:
        pixels = src.read()
        transform, crs = src.transform, src.crs.to_string()
    if pixels.shape[0] != len(scene.wavelengths_nm):
        raise ValueError(f"{scene.scene_id}: {pixels.shape[0]} bands in the cube, "
                         f"{len(scene.wavelengths_nm)} in METADATA.XML")
    with rasterio.open(find_member(scene.folder, "QL_PIXELMASK")) as src:
        flagged = src.read() != 0                                 # 0 = valid, else flagged
    pixels[flagged] = NODATA
    labels = {}
    for key, stem in LABEL_LAYERS.items():
        with rasterio.open(find_member(scene.folder, stem)) as src:
            labels[key] = src.read(1)
    return SceneArrays(pixels, labels, transform, crs)


def tile_grid(height: int, width: int, tile: int = TILE) -> list[tuple[int, int]]:
    """(row, col) of every full tile, row-major. Partial edge strips are left out."""
    return [(r, c) for r in range(height // tile) for c in range(width // tile)]


def tiles(scene: EnmapScene, arrays: SceneArrays | None = None,
          min_valid_fraction: float = MIN_VALID_FRACTION) -> Iterator[tuple[str, dict]]:
    """(key, sample) for every kept tile. key = `en_<datatake>_<tile>_r<row>c<col>`."""
    a = arrays or read_scene(scene)
    facts = scene.facts
    wavelengths = scene.wavelengths_nm.astype(np.float32)
    _, h, w = a.pixels.shape
    for r, c in tile_grid(h, w):
        y0, x0 = r * TILE, c * TILE
        px = a.pixels[:, y0:y0 + TILE, x0:x0 + TILE]
        valid = float((px != NODATA).any(axis=0).mean())
        if valid < min_valid_fraction:
            continue
        t = a.transform * rasterio.Affine.translation(x0, y0)
        cx, cy = t * (TILE / 2, TILE / 2)
        (lon,), (lat,) = warp_transform(a.crs, "EPSG:4326", [cx], [cy])
        meta = {**facts, "row": r, "col": c, "y0": y0, "x0": x0, "valid_fraction": round(valid, 4),
                "lon": round(lon, 6), "lat": round(lat, 6), "crs": a.crs, "transform": list(t)[:6]}
        sample = {"pixels.npy": np.ascontiguousarray(px), "wavelengths.npy": wavelengths, "meta.json": meta}
        for key, layer in a.labels.items():
            sample[f"{key}.npy"] = np.ascontiguousarray(layer[None, y0:y0 + TILE, x0:x0 + TILE])
        # No dots in the key: webdataset splits the extension at the first one.
        yield f"en_{facts['datatake']}_{facts['tile']}_r{r}c{c}", sample
