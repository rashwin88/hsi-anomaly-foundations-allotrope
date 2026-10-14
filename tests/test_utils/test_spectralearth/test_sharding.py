"""SpectralEarth sharding: plan, per-band statistics, shard bytes."""

import io
import tarfile

import numpy as np
import pytest

from app.models.dataset.spectralearth import SpectralEarthPatch, load_band_table
from app.utils.spectralearth.catalog import CatalogEntry
from app.utils.spectralearth.sharding import BandStats, shard_plan, write_shard


def _entry(i: int) -> CatalogEntry:
    return CatalogEntry(path=f"enmap/{i:07d}/00001.tif", site_id=f"{i:07d}", patch_id="00001",
                        size_bytes=1, sha256="0" * 64, bucket="b", key=f"k{i}")


def _patch(i: int) -> SpectralEarthPatch:
    raw = np.random.default_rng(i).integers(0, 5000, (202, 128, 128), dtype=np.int16)
    return SpectralEarthPatch(f"{i:07d}", "00001", raw, load_band_table())


def test_plan_is_deterministic_complete_and_shuffled():
    entries = [_entry(i) for i in range(1000)]
    plan = shard_plan(entries, seed=7, per_shard=150)
    assert [len(s) for s in plan] == [150] * 6 + [100]
    assert sorted(e.path for s in plan for e in s) == sorted(e.path for e in entries)
    assert plan == shard_plan(list(reversed(entries)), seed=7, per_shard=150)   # input order irrelevant
    assert plan != shard_plan(entries, seed=8, per_shard=150)
    assert [e.path for e in plan[0]] != sorted(e.path for e in plan[0])         # not path order


def test_band_stats_match_numpy_and_skip_nodata():
    raws = [_patch(i).raw for i in range(3)]
    raws[0][4, :10, :10] = -32768
    stats = BandStats()
    for r in raws:
        stats.add(r, 0.0001, -32768)
    cube = np.stack(raws).astype(np.float64)
    band4 = cube[:, 4][cube[:, 4] != -32768] * 0.0001
    mean, std = stats.mean_std()
    assert stats.count[4] == 3 * 128 * 128 - 100
    assert mean[4] == pytest.approx(band4.mean(), rel=1e-12)
    assert std[4] == pytest.approx(band4.std(), rel=1e-9)


def test_band_stats_merge_equals_one_pass():
    a, b, both = BandStats(), BandStats(), BandStats()
    for i, s in ((0, a), (1, b)):
        s.add(_patch(i).raw, 0.0001, -32768)
        both.add(_patch(i).raw, 0.0001, -32768)
    merged = a.merge(b)
    assert np.array_equal(merged.count, both.count)
    assert np.allclose(merged.total, both.total) and np.allclose(merged.total_sq, both.total_sq)
    assert np.array_equal(BandStats().merge(a).count, a.count)
    assert np.array_equal(a.merge(BandStats()).count, a.count)


def test_band_stats_refuse_an_empty_band():
    with pytest.raises(ValueError):
        BandStats().mean_std()


def test_shard_bytes_are_deterministic_and_webdataset_shaped():
    patches = [_patch(i) for i in range(2)]
    first, second = io.BytesIO(), io.BytesIO()
    keys, stats = write_shard(patches, first)
    write_shard(patches, second)
    assert first.getvalue() == second.getvalue()
    assert keys == ["se_0000000_00001", "se_0000001_00001"]
    assert int(stats.count[0]) == 2 * 128 * 128
    with tarfile.open(fileobj=io.BytesIO(first.getvalue())) as tar:
        names = tar.getnames()
        pixels = np.load(io.BytesIO(tar.extractfile(names[0]).read()))
        wl = np.load(io.BytesIO(tar.extractfile(names[1]).read()))
    assert names == ["se_0000000_00001.pixels.npy", "se_0000000_00001.wavelengths.npy",
                     "se_0000001_00001.pixels.npy", "se_0000001_00001.wavelengths.npy"]
    assert pixels.dtype == np.int16 and np.array_equal(pixels, patches[0].raw)
    assert wl.dtype == np.float32 and wl.shape == (202,)


def test_shard_reads_back_through_webdataset(tmp_path):
    wds = pytest.importorskip("webdataset")
    patches = [_patch(i) for i in range(2)]
    path = tmp_path / "shard-000000.tar"
    with path.open("wb") as f:
        write_shard(patches, f)
    # "file:" + forward slashes: webdataset reads a bare "C:\..." as a URL with scheme "c"
    samples = list(wds.WebDataset(["file:" + path.as_posix()], shardshuffle=False).decode())
    assert [s["__key__"] for s in samples] == [p.key for p in patches]
    assert np.array_equal(samples[1]["pixels.npy"], patches[1].raw)


def test_recompress_shard_is_exact_and_deterministic():
    from app.utils.shard_codec import decode_i16dz
    from app.utils.spectralearth.sharding import recompress_shard
    patches = [_patch(i) for i in range(2)]
    patches[0].raw[:, :10, :10] = -32768
    raw = io.BytesIO()
    write_shard(patches, raw)
    first, second = io.BytesIO(), io.BytesIO()
    keys = recompress_shard(io.BytesIO(raw.getvalue()), first)
    recompress_shard(io.BytesIO(raw.getvalue()), second)
    assert keys == [p.key for p in patches]
    assert first.getvalue() == second.getvalue()
    with tarfile.open(fileobj=io.BytesIO(first.getvalue())) as tar:
        names = tar.getnames()
        assert names == ["se_0000000_00001.pixels.i16dz", "se_0000000_00001.wavelengths.npy",
                         "se_0000001_00001.pixels.i16dz", "se_0000001_00001.wavelengths.npy"]
        for p, name in zip(patches, names[::2]):
            assert np.array_equal(decode_i16dz(tar.extractfile(name).read()), p.raw)
