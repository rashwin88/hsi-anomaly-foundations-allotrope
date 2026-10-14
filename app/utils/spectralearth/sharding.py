"""
Pack SpectralEarth patches into WebDataset shards, and measure per-band statistics on the way.

Driven by scripts/shard_spectralearth.py; the shards are read by the trainer's hot_storage
path (app/abstract_classes/foundation_trainer.py) like any other shard set.

Training cannot read patches one S3 object at a time - that manages ~18 patches/s against the
hundreds a GPU needs - so patches are packed into ~1 GB tars that are copied to local disk
once. The order is shuffled across the whole collection before packing, because the trainer
only shuffles shard order plus a small buffer: packed in path order, a shard would hold
neighbouring patches of one scene and every batch would look alike.

Gotchas:
  - A shard is a fixed number of patches, not a size target, so shard n always holds the
    same patches for a given seed. That is what makes an interrupted run resumable.
  - pixels.npy is the raw int16 (202, 128, 128), x 0.0001 = reflectance, -32768 = nodata;
    the trainer converts. Stored raw it is half the size of float32.
"""

import io
import random
import tarfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from app.models.dataset.spectralearth import SpectralEarthPatch
from app.utils.spectralearth.catalog import CatalogEntry

# 150 x 6.62 MB (202 x 128 x 128 int16 + .npy header) = 0.99 GB, the ~1 GB shard size the
# existing patchers use (TARGET_GB in app/utils/patch_generation/final/).
PATCHES_PER_SHARD = 150


def shard_plan(entries: Sequence[CatalogEntry], seed: int, per_shard: int = PATCHES_PER_SHARD
               ) -> list[list[CatalogEntry]]:
    """Entries shuffled with `seed`, then cut into consecutive shards of `per_shard`
    (the last one shorter). Same input and seed -> same plan on any machine."""
    order = sorted(entries, key=lambda e: e.path)
    random.Random(seed).shuffle(order)
    return [order[i:i + per_shard] for i in range(0, len(order), per_shard)]


@dataclass
class BandStats:
    """Running per-band count, sum and sum of squares of valid reflectance.

    The raw material for wavelength normalisation (docs/proposals/spectral-attention-
    bottleneck.md, decision 5). Mergeable, so each shard records its own and a resumed
    run adds them up without re-reading a patch.
    """

    count: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    total: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))
    total_sq: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float64))

    def add(self, raw: np.ndarray, scale: float, nodata: int) -> None:
        """Fold in one (bands, rows, cols) int16 patch; nodata voxels are skipped."""
        valid = raw != nodata
        # float64: reflectance ~0.2 with variance ~0.01 over 10^10 voxels - float32 sums
        # would lose the variance to rounding.
        r = np.where(valid, raw.astype(np.float64) * scale, 0.0)
        n, s, sq = valid.sum(axis=(1, 2)), r.sum(axis=(1, 2)), (r * r).sum(axis=(1, 2))
        if self.count.size == 0:
            self.count, self.total, self.total_sq = n.astype(np.int64), s, sq
        else:
            self.count, self.total, self.total_sq = self.count + n, self.total + s, self.total_sq + sq

    def merge(self, other: "BandStats") -> "BandStats":
        if self.count.size == 0 or other.count.size == 0:
            src = other if self.count.size == 0 else self
            return BandStats(src.count.copy(), src.total.copy(), src.total_sq.copy())
        return BandStats(self.count + other.count, self.total + other.total, self.total_sq + other.total_sq)

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-band mean and population std of reflectance. Raises ValueError if a band
        has no valid voxel."""
        if self.count.size == 0 or (self.count == 0).any():
            raise ValueError("a band has no valid voxels")
        mean = self.total / self.count
        return mean, np.sqrt(np.maximum(self.total_sq / self.count - mean * mean, 0.0))


def _npy(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def write_shard(patches: Iterable[SpectralEarthPatch], fileobj) -> tuple[list[str], BandStats]:
    """Write patches as one WebDataset tar to a binary file object, in the order given.

    Per sample: `<key>.pixels.npy` (raw int16) and `<key>.wavelengths.npy` (float32, band
    order). Returns the keys written and the shard's BandStats. Members carry mtime 0, so
    the same patches always produce byte-identical tars.
    """
    keys, stats = [], BandStats()
    with tarfile.open(fileobj=fileobj, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for patch in patches:
            for name, data in (("pixels.npy", _npy(patch.raw)),
                               ("wavelengths.npy", _npy(patch.wavelengths_nm.astype(np.float32)))):
                info = tarfile.TarInfo(f"{patch.key}.{name}")
                info.size, info.mtime, info.mode = len(data), 0, 0o444
                tar.addfile(info, io.BytesIO(data))
            stats.add(patch.raw, patch.bands.reflectance_scale, patch.bands.nodata)
            keys.append(patch.key)
    return keys, stats
