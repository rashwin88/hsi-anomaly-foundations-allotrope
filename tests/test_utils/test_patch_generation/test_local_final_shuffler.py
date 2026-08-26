"""
Tests for the segmentation lane's final shuffle stage.

Synthetic shards, no payloads, no AWS — runs anywhere.

The stage exists because the intermediate sharder writes one shard per scene
(for resume), which leaves each shard holding a few hundred spatially
sequential tiles of one scene. Two properties matter and both are asserted
here: **nothing is lost or duplicated**, and **the output is actually mixed**.
A shuffler that silently dropped a tenth of the data would look fine.
"""

import os
import tarfile

import numpy as np
import pytest
import webdataset as wds

from app.utils.patch_generation.final.local_final_shuffler import LocalFinalShuffler

SCENES, PER_SCENE = 6, 30
TOTAL = SCENES * PER_SCENE


@pytest.fixture
def sources(tmp_path):
    """One shard per fake scene, each holding sequential patches."""
    src = tmp_path / "intermediate"
    src.mkdir()
    for s in range(SCENES):
        with open(src / f"enmap_seg_scene{s}.tar", "wb") as fh, wds.TarWriter(fh) as w:
            for p in range(PER_SCENE):
                w.write({
                    "__key__": f"scene{s}#row_coord:{p}#col_coord:0",
                    "pixels.npy": np.full((2, 4, 4), s, np.float16),
                    "label_cloud.npy": np.full((1, 4, 4), p % 2, np.uint8),
                    "meta.json": {"sensor": "enmap", "scene": s},
                })
    return str(src)


def _read(dest):
    """(keys in stream order, scene index per key, set of extensions)."""
    keys, scenes, exts = [], [], set()
    for name in sorted(f for f in os.listdir(dest) if f.endswith(".tar")):
        with tarfile.open(os.path.join(dest, name)) as tf:
            for member in tf.getnames():
                key, _, ext = member.partition(".")
                exts.add(ext)
                if ext == "pixels.npy":
                    keys.append(key)
                    scenes.append(int(key[5]))
    return keys, scenes, exts


def _shuffle(sources, tmp_path, name="final", **kwargs):
    dest = tmp_path / name
    opts = dict(shuffle_size=40, group_size=SCENES, shard_size_bytes=40_000)
    opts.update(kwargs)
    LocalFinalShuffler(source_dir=sources, dest_dir=str(dest), **opts).write_shards()
    return str(dest)


def test_nothing_is_lost(sources, tmp_path):
    keys, _, _ = _read(_shuffle(sources, tmp_path))
    assert len(keys) == TOTAL


def test_nothing_is_duplicated(sources, tmp_path):
    keys, _, _ = _read(_shuffle(sources, tmp_path))
    assert len(set(keys)) == TOTAL, "one complete pass — resampled=False"


def test_every_key_type_survives(sources, tmp_path):
    _, _, exts = _read(_shuffle(sources, tmp_path))
    assert exts == {"pixels.npy", "label_cloud.npy", "meta.json"}


def test_payload_bytes_are_unchanged(sources, tmp_path):
    """Records move as opaque bytes — no decode/re-encode round trip."""
    dest = _shuffle(sources, tmp_path)

    def payloads(folder):
        out = {}
        for name in sorted(f for f in os.listdir(folder) if f.endswith(".tar")):
            with tarfile.open(os.path.join(folder, name)) as tf:
                for m in tf.getmembers():
                    if m.name.endswith("pixels.npy"):
                        out[m.name] = tf.extractfile(m).read()
        return out

    before, after = payloads(sources), payloads(dest)
    assert set(before) == set(after)
    assert all(before[k] == after[k] for k in before)


def test_output_is_actually_mixed(sources, tmp_path):
    """Input is 6 runs of 30. Mixed output has neighbours from other scenes."""
    _, scenes, _ = _read(_shuffle(sources, tmp_path))
    changes = sum(1 for a, b in zip(scenes, scenes[1:]) if a != b)
    assert changes > 0.5 * (TOTAL - 1), f"only {changes}/{TOTAL - 1} — barely shuffled"


def test_same_seed_reproduces(sources, tmp_path):
    a, _, _ = _read(_shuffle(sources, tmp_path, name="a", seed=7))
    b, _, _ = _read(_shuffle(sources, tmp_path, name="b", seed=7))
    assert a == b


def test_different_seed_reorders(sources, tmp_path):
    a, _, _ = _read(_shuffle(sources, tmp_path, name="a", seed=7))
    b, _, _ = _read(_shuffle(sources, tmp_path, name="b", seed=8))
    assert a != b and set(a) == set(b)


def test_rolls_into_multiple_shards(sources, tmp_path):
    dest = _shuffle(sources, tmp_path, shard_size_bytes=20_000)
    assert len([f for f in os.listdir(dest) if f.endswith(".tar")]) > 1


def test_one_shard_when_cap_is_large(sources, tmp_path):
    dest = _shuffle(sources, tmp_path, shard_size_bytes=1 << 30)
    assert len([f for f in os.listdir(dest) if f.endswith(".tar")]) == 1


def test_inputs_are_left_alone(sources, tmp_path):
    before = sorted(os.listdir(sources))
    _shuffle(sources, tmp_path)
    assert sorted(os.listdir(sources)) == before, "verify output before deleting inputs"


def test_empty_source_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        LocalFinalShuffler(source_dir=str(empty), dest_dir=str(tmp_path / "out"))
