"""
Every SpectralEarth patch we hold, as an in-memory index with a stable train/test split.

Read by app/utils/spectralearth/source.py (to fetch and verify bytes) and by the torch
dataset (to decide what an epoch contains). Built from the inventory collection's
`files.jsonl.gz` — one line per patch, written and verified by the inventory when the
archive was unpacked — so it costs one S3 GET instead of listing 539,755 objects, and every
entry carries the SHA-256 its bytes must match.

The split is by *site*, not by patch: about 17% of SpectralEarth sites hold several dates
of the same place, and a site whose dates straddle train and test would leak that place
into evaluation. A site's side is decided by hashing `"<seed>:<site_id>"`, so it is the
same on every machine and run for a given seed, and adding or removing patches never moves
an existing site across the split.

Gotcha: `app/` must not import the inventory service, so this reads the bucket layout
directly — `<collection prefix>/files.jsonl.gz` and `<collection prefix>/payload/<path>`.
"""

import gzip
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from app.models.dataset.spectralearth import parse_member_path
from app.utils.general_utils import s3_config


@dataclass(frozen=True)
class CatalogEntry:
    """One patch: where it is and what its bytes must hash to."""

    path: str            # `enmap/<site>/<patch>.tif`, relative to the collection payload
    site_id: str
    patch_id: str
    size_bytes: int
    sha256: str
    bucket: str
    key: str             # full S3 key of the object


def _site_fraction(seed: str, site_id: str) -> float:
    """Uniform in [0, 1), fixed for a (seed, site) pair."""
    digest = hashlib.sha256(f"{seed}:{site_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


class SpectralEarthCatalog:
    """An ordered, immutable set of CatalogEntry; operations return new catalogues."""

    def __init__(self, entries: Iterable[CatalogEntry]):
        self._entries = sorted(entries, key=lambda e: e.path)

    @classmethod
    def from_manifest_lines(cls, lines: Iterable[str], bucket: str, prefix: str) -> "SpectralEarthCatalog":
        """Entries from `files.jsonl.gz` lines; non-patch files are skipped."""
        prefix = prefix.rstrip("/") + "/"
        out = []
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row["relative_path"].endswith(".tif"):
                continue
            site_id, patch_id = parse_member_path(row["relative_path"])
            out.append(CatalogEntry(
                path=row["relative_path"], site_id=site_id, patch_id=patch_id,
                size_bytes=int(row["size_bytes"]), sha256=row["sha256"],
                bucket=bucket, key=f"{prefix}payload/{row['relative_path']}",
            ))
        return cls(out)

    @classmethod
    def from_files_manifest(cls, collection_uri: str, s3_client=None) -> "SpectralEarthCatalog":
        """Load from the collection's prefix: `s3://bucket/<prefix>/` or a local folder
        holding `files.jsonl.gz` (tests, offline copies)."""
        if collection_uri.startswith("s3://"):
            bucket, _, prefix = collection_uri.removeprefix("s3://").partition("/")
            prefix = prefix.rstrip("/")
            client = s3_client or s3_config.client()
            body = client.get_object(Bucket=bucket, Key=f"{prefix}/files.jsonl.gz")["Body"].read()
        else:
            bucket, prefix = "", collection_uri.rstrip("/")
            body = (Path(collection_uri) / "files.jsonl.gz").read_bytes()
        return cls.from_manifest_lines(gzip.decompress(body).decode().splitlines(), bucket, prefix)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self._entries)

    def __getitem__(self, i: int) -> CatalogEntry:
        return self._entries[i]

    @property
    def total_bytes(self) -> int:
        return sum(e.size_bytes for e in self._entries)

    def by_site(self) -> dict[str, list[CatalogEntry]]:
        """Site id -> its patches (several when the site was imaged on several dates)."""
        sites: dict[str, list[CatalogEntry]] = defaultdict(list)
        for e in self._entries:
            sites[e.site_id].append(e)
        return dict(sites)

    def filter(self, keep: Callable[[CatalogEntry], bool]) -> "SpectralEarthCatalog":
        return SpectralEarthCatalog(e for e in self._entries if keep(e))

    def split(self, test_fraction: float, seed: str) -> tuple["SpectralEarthCatalog", "SpectralEarthCatalog"]:
        """(train, test), with every site wholly on one side. Raises ValueError unless
        0 < test_fraction < 1."""
        if not 0 < test_fraction < 1:
            raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
        test = self.filter(lambda e: _site_fraction(seed, e.site_id) < test_fraction)
        train = self.filter(lambda e: _site_fraction(seed, e.site_id) >= test_fraction)
        return train, test

    def sample(self, n: int, seed: int = 0) -> "SpectralEarthCatalog":
        """n patches chosen reproducibly (all of them when n >= len)."""
        if n >= len(self._entries):
            return SpectralEarthCatalog(self._entries)
        return SpectralEarthCatalog(random.Random(seed).sample(self._entries, n))
