"""
Fetch SpectralEarth patches by catalogue entry, verified, optionally through a disk cache.

Sits between SpectralEarthCatalog (what to read) and SpectralEarthPatch (what the bytes
mean); the torch dataset and the sanity tools call load()/load_many().

Every patch is checked against the SHA-256 the inventory recorded before it is decoded or
cached, whether it came from S3 or from the cache. Training reads the same patch many times
over many epochs, often from a local copy that outlives the run; without the check a
truncated download or a flipped bit on a cache disk would feed silently wrong spectra into
the model. Hashing 6.6 MB costs a few milliseconds against tens for the GET.

Gotchas:
  - The cache is keyed by the patch's path, so one cache folder serves one collection.
  - load_many() returns patches in the order given, fetching up to `workers` at once.
"""

import hashlib
import os
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app.models.dataset.spectralearth import SpectralEarthPatch
from app.utils.general_utils import s3_config
from app.utils.spectralearth.catalog import CatalogEntry


class PatchIntegrityError(ValueError):
    """Bytes for a patch didn't match the SHA-256 the inventory recorded."""


class SpectralEarthSource:
    def __init__(self, cache_dir: str | os.PathLike | None = None, s3_client=None, workers: int = 32):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.workers = workers
        self._client = s3_client

    @property
    def client(self):
        if self._client is None:
            self._client = s3_config.client()
        return self._client

    def fetch_bytes(self, entry: CatalogEntry) -> bytes:
        """The patch's verified bytes. Raises PatchIntegrityError on a mismatch from S3;
        a mismatching cache file is discarded and re-fetched."""
        cached = self.cache_dir / entry.path if self.cache_dir else None
        if cached is not None and cached.exists():
            data = cached.read_bytes()
            if hashlib.sha256(data).hexdigest() == entry.sha256:
                return data
            cached.unlink()
        data = self.client.get_object(Bucket=entry.bucket, Key=entry.key)["Body"].read()
        if hashlib.sha256(data).hexdigest() != entry.sha256:
            raise PatchIntegrityError(f"{entry.path}: S3 bytes don't match the inventory's sha256")
        if cached is not None:
            # Write-then-rename, so an interrupted write never leaves a truncated file
            # that would be read back as a cache hit.
            cached.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=cached.parent, suffix=".part")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, cached)
        return data

    def load(self, entry: CatalogEntry) -> SpectralEarthPatch:
        return SpectralEarthPatch.from_geotiff_bytes(self.fetch_bytes(entry), entry.path)

    def load_many(self, entries: Iterable[CatalogEntry]) -> Iterator[SpectralEarthPatch]:
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            yield from pool.map(self.load, entries)
