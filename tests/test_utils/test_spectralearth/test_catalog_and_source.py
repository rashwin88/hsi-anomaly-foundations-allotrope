"""SpectralEarth catalogue (split, grouping, sampling) and source (verify, cache)."""

import gzip
import hashlib
import io
import json

import numpy as np
import pytest
from rasterio.io import MemoryFile

from app.utils.spectralearth.catalog import SpectralEarthCatalog
from app.utils.spectralearth.source import PatchIntegrityError, SpectralEarthSource


def _tif(seed: int) -> bytes:
    cube = np.random.default_rng(seed).integers(0, 5000, (202, 128, 128), dtype=np.int16)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", count=202, height=128, width=128, dtype="int16", nodata=-32768) as d:
            d.write(cube)
        return mem.read()


@pytest.fixture
def collection(tmp_path):
    """A local collection: 400 sites, sites divisible by 4 hold two dates."""
    rows = []
    for s in range(400):
        for p in range(2 if s % 4 == 0 else 1):
            path = f"enmap/{s:07d}/{p:05d}.tif"
            rows.append({"relative_path": path, "size_bytes": 10, "sha256": "0" * 64,
                         "part_sha256": [], "s3_version_id": None})
    rows.append({"relative_path": "README.txt", "size_bytes": 1, "sha256": "0" * 64,
                 "part_sha256": [], "s3_version_id": None})
    (tmp_path / "files.jsonl.gz").write_bytes(
        gzip.compress("\n".join(json.dumps(r) for r in rows).encode()))
    return tmp_path


def test_manifest_loads_patches_only(collection):
    cat = SpectralEarthCatalog.from_files_manifest(str(collection))
    assert len(cat) == 500 and len(cat.by_site()) == 400
    assert all(e.path.endswith(".tif") for e in cat)
    assert cat[0].key.endswith(f"/payload/{cat[0].path}")


def test_split_is_by_site_stable_and_roughly_sized(collection):
    cat = SpectralEarthCatalog.from_files_manifest(str(collection))
    train, test = cat.split(0.25, seed="se-v1")
    train_sites = {e.site_id for e in train}
    test_sites = {e.site_id for e in test}
    assert not train_sites & test_sites
    assert len(train) + len(test) == len(cat)
    assert 0.15 < len(test_sites) / 400 < 0.35
    again_train, again_test = cat.split(0.25, seed="se-v1")
    assert [e.path for e in again_test] == [e.path for e in test]
    other_train, other_test = cat.split(0.25, seed="se-v2")
    assert {e.site_id for e in other_test} != test_sites


def test_split_does_not_move_sites_when_patches_are_removed(collection):
    cat = SpectralEarthCatalog.from_files_manifest(str(collection))
    _, test = cat.split(0.25, seed="se-v1")
    _, test_subset = cat.filter(lambda e: int(e.site_id) % 2 == 0).split(0.25, seed="se-v1")
    assert {e.site_id for e in test_subset} <= {e.site_id for e in test}


def test_split_rejects_bad_fraction(collection):
    with pytest.raises(ValueError):
        SpectralEarthCatalog.from_files_manifest(str(collection)).split(1.0, seed="x")


def test_sample_is_reproducible(collection):
    cat = SpectralEarthCatalog.from_files_manifest(str(collection))
    assert [e.path for e in cat.sample(50, seed=3)] == [e.path for e in cat.sample(50, seed=3)]
    assert len(cat.sample(10_000)) == len(cat)


class _FakeS3:
    """Serves fixed bytes per key and counts GETs."""

    def __init__(self, blobs: dict[str, bytes]):
        self.blobs, self.gets = blobs, 0

    def get_object(self, Bucket, Key):
        self.gets += 1
        return {"Body": io.BytesIO(self.blobs[Key])}


def _entry_catalog(blobs: dict[str, bytes], sha_override: dict[str, str] | None = None):
    lines = [json.dumps({"relative_path": p, "size_bytes": len(b),
                         "sha256": (sha_override or {}).get(p, hashlib.sha256(b).hexdigest())})
             for p, b in blobs.items()]
    return SpectralEarthCatalog.from_manifest_lines(lines, "bkt", "coll")


def test_source_verifies_decodes_and_caches(tmp_path):
    blobs = {f"enmap/000000{i}/0000{i}.tif": _tif(i) for i in range(3)}
    cat = _entry_catalog(blobs)
    s3 = _FakeS3({e.key: blobs[e.path] for e in cat})
    src = SpectralEarthSource(cache_dir=tmp_path / "cache", s3_client=s3, workers=2)
    patches = list(src.load_many(cat))
    assert [p.site_id for p in patches] == [e.site_id for e in cat]          # order kept
    assert patches[0].raw.shape == (202, 128, 128)
    assert s3.gets == 3
    list(src.load_many(cat))
    assert s3.gets == 3                                                       # second pass from cache


def test_source_refetches_a_corrupted_cache_file(tmp_path):
    blobs = {"enmap/0000001/00001.tif": _tif(1)}
    cat = _entry_catalog(blobs)
    s3 = _FakeS3({e.key: blobs[e.path] for e in cat})
    src = SpectralEarthSource(cache_dir=tmp_path, s3_client=s3)
    src.load(cat[0])
    (tmp_path / cat[0].path).write_bytes(b"corrupt")
    assert src.load(cat[0]).site_id == "0000001" and s3.gets == 2


def test_source_rejects_bytes_that_dont_match_the_inventory():
    blobs = {"enmap/0000001/00001.tif": _tif(1)}
    cat = _entry_catalog(blobs, sha_override={"enmap/0000001/00001.tif": "f" * 64})
    src = SpectralEarthSource(s3_client=_FakeS3({e.key: blobs[e.path] for e in cat}))
    with pytest.raises(PatchIntegrityError):
        src.load(cat[0])
