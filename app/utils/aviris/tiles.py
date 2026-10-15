"""
AVIRIS-NG flight lines cut into 128 px training tiles at native bands and native pixel size - the
airborne part of the fine-tuning shards (docs/lld/aviris-shards.md).

Used by scripts/shard_aviris.py. Same layout as the EnMAP and PRISMA shards except that pixels are
stored compressed - `pixels.i16dz` (app/utils/shard_codec.py: exact, ~1.9x smaller) - because 372
bands x 128 x 128 int16 is 12 MB a tile and the campaign is ~20,000 tiles. The spectral-stem
trainer decodes `.i16dz` in its loader workers. Plus wavelengths.npy float32 (372,) and meta.json.
No label keys: AVIRIS-NG L2 carries no cloud or water mask.

Gotchas:
  - Tiles are at the flight line's own pixel size (3.2-8.2 m), not resampled to 30 m: fine-tuning
    sees airborne texture alongside EnMAP and PRISMA (decided 2026-10-15).
  - Per-band stats are accumulated here, from the int16 tile before compression, into the
    BandStats passed in - write_samples only counts uncompressed `pixels.npy`.
"""

from collections.abc import Iterator

import numpy as np

from app.utils.aviris.scene import NODATA, REFLECTANCE_SCALE, AvirisScene
from app.utils.enmap.tiles import MIN_VALID_FRACTION, TILE
from app.utils.shard_codec import encode_i16dz
from app.utils.spectralearth.sharding import BandStats


def tiles(scene: AvirisScene, stats: BandStats | None = None,
          min_valid_fraction: float = MIN_VALID_FRACTION) -> Iterator[tuple[str, dict]]:
    """(key, sample) for every kept full tile, strip by strip. key = `av_<yyyymmdd>t<hhmmss>_r<row>c<col>`."""
    stamp = scene.scene_id[3:]
    for r in range(scene.lines // TILE):
        strip = scene.strip(r * TILE, TILE)                     # (C, 128, samples)
        for c in range(scene.samples // TILE):
            x0 = c * TILE
            px = np.ascontiguousarray(strip[:, :, x0:x0 + TILE])
            valid = float((px != NODATA).any(axis=0).mean())
            if valid < min_valid_fraction:
                continue
            if stats is not None:
                stats.add(px, REFLECTANCE_SCALE, NODATA)
            lon, lat = scene.lonlat(x0 + TILE / 2, r * TILE + TILE / 2)
            meta = {**{k: v for k, v in scene.facts.items() if k != "map"}, "row": r, "col": c,
                    "y0": r * TILE, "x0": x0, "valid_fraction": round(valid, 4), "lat": round(lat, 6), "lon": round(lon, 6)}
            yield f"av_{stamp}_r{r}c{c}", {"pixels.i16dz": encode_i16dz(px), "wavelengths.npy": scene.wavelengths_nm,
                                            "meta.json": meta}
