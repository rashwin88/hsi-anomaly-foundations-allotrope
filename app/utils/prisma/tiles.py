"""
PRISMA scenes cut into 128 px training tiles at native bands - the PRISMA half of the
fine-tuning shards (docs/lld/prisma-shards.md).

Used by scripts/shard_prisma.py. Same sample layout as the EnMAP shards (app/utils/enmap/tiles.py)
so the spectral-stem trainer reads both unchanged: pixels.npy int16 (234, 128, 128), x 0.0001 =
reflectance, -32768 = nodata; wavelengths.npy float32 (234,); meta.json. No label keys: PRISMA L2D
carries no cloud or water mask - the segmentation head learns from EnMAP, and PRISMA batches add
reconstruction only.

Gotchas:
  - Partial edge strips are dropped, as for EnMAP (1186 x 1196 gives 9 x 9 full tiles).
  - lat/lon in meta.json come from the scene's per-pixel geolocation at the tile centre; L2D is
    map-projected (Epsg_Code, kept in meta) but the HE5 carries no affine, so none is stored.
"""

from collections.abc import Iterator

import numpy as np

from app.utils.enmap.tiles import MIN_VALID_FRACTION, TILE, tile_grid
from app.utils.prisma.scene import NODATA, PrismaScene


def tiles(scene: PrismaScene, min_valid_fraction: float = MIN_VALID_FRACTION) -> Iterator[tuple[str, dict]]:
    """(key, sample) for every kept tile. key = `pr_<acquisition start>_r<row>c<col>` (no dots:
    webdataset splits the extension at the first one)."""
    _, h, w = scene.pixels.shape
    start = scene.scene_id.split("_")[3]                       # PRS_L2D_STD_<start>_<stop>_0001
    for r, c in tile_grid(h, w):
        y0, x0 = r * TILE, c * TILE
        px = scene.pixels[:, y0:y0 + TILE, x0:x0 + TILE]
        valid = float((px != NODATA).any(axis=0).mean())
        if valid < min_valid_fraction:
            continue
        cy, cx = y0 + TILE // 2, x0 + TILE // 2
        meta = {**scene.facts, "row": r, "col": c, "y0": y0, "x0": x0, "valid_fraction": round(valid, 4),
                "lat": round(float(scene.lat[cy, cx]), 6), "lon": round(float(scene.lon[cy, cx]), 6)}
        yield f"pr_{start}_r{r}c{c}", {"pixels.npy": np.ascontiguousarray(px),
                                         "wavelengths.npy": scene.wavelengths_nm, "meta.json": meta}
