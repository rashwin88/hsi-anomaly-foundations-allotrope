"""
Tests for the three intermediate sharders.

These are the first tests these classes have ever had, and the reason is
recorded in `docs/lld/segmentation-sharding.md`: scene discovery used
to run inside `__init__`, so constructing one required a network and AWS
credentials. Nothing here mocks S3 — it asserts that S3 is never reached.

No payloads, no AWS, no marker: these run everywhere.
"""

import random

import pytest

from app.utils.patch_generation.intermediate.enmap_intermediate_patcher import (
    EnmapIntermediateSharder,
)
from app.utils.patch_generation.intermediate.landsat_intermediate_patcher import (
    LandsatIntermediateSharder,
)
from app.utils.patch_generation.intermediate.prisma_intermediate_patcher import (
    PrismaIntermediateSharder,
)

# Each sharder, the attribute holding its split, and plausible scene ids.
# PRISMA lists .he5 files; the other two list folder prefixes.
#
# The ids are deliberately NOT in sorted order. The split does
# `sorted(...)` before the seeded shuffle so that the seed, not the listing
# order, decides the split — and S3 listing order is not guaranteed. With a
# pre-sorted fixture that sorted() is a no-op and the tests below silently
# stop guarding it (confirmed by mutation: removing sorted() still passed).
_SCRAMBLE = [7, 2, 9, 0, 4, 1, 8, 3, 6, 5]
SHARDERS = [
    (EnmapIntermediateSharder, "_scene_prefixes", [f"enmap/EN_{i:02d}/" for i in _SCRAMBLE]),
    (PrismaIntermediateSharder, "_scene_keys", [f"prisma/PRS_{i:02d}.he5" for i in _SCRAMBLE]),
    (LandsatIntermediateSharder, "_scene_prefixes", [f"landsat/LC08_{i:02d}/" for i in _SCRAMBLE]),
]
IDS = [cls.__name__ for cls, _, _ in SHARDERS]


@pytest.fixture
def counted(monkeypatch):
    """Replace list_scenes with a counter, so any call to S3 is visible."""

    def _install(cls, scenes):
        calls = []

        def fake(self):
            calls.append(1)
            return list(scenes)

        monkeypatch.setattr(cls, "list_scenes", fake)
        return calls

    return _install


def _build(cls, split="train", **kwargs):
    return cls(
        source_folder="/tmp/src/",
        destination_folder="/tmp/dst/",
        split=split,
        **kwargs,
    )


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_construction_touches_no_network(cls, attr, scenes, counted):
    """The whole point of the lazy-discovery change. If this regresses, these
    classes become untestable again and the debt comes back."""
    calls = counted(cls, scenes)
    _build(cls)
    assert calls == [], "scene discovery must not happen in __init__"


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_discovery_happens_once_on_first_access(cls, attr, scenes, counted):
    calls = counted(cls, scenes)
    sharder = _build(cls)
    getattr(sharder, attr)
    assert len(calls) == 1
    getattr(sharder, attr)
    assert len(calls) == 1, "result must be cached, not re-fetched"


def _expected(scenes, seed=42, test_fraction=0.2, max_scenes=None, split="train"):
    """A hand-rolled replay of the original inline split, so the tests below
    check the algorithm rather than checking the code against itself."""
    ordered = sorted(scenes)
    random.Random(seed).shuffle(ordered)
    if max_scenes is not None and max_scenes < len(ordered):
        ordered = ordered[:max_scenes]
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered[:cut] if split == "train" else ordered[cut:]


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_split_matches_the_original_algorithm(cls, attr, scenes, counted):
    """sorted -> seeded shuffle -> cut. Chunks that inject storage must not
    perturb which scenes land in which split."""
    counted(cls, scenes)
    for split in ("train", "test"):
        got = getattr(_build(cls, split=split), attr)
        assert got == _expected(scenes, split=split)


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_train_and_test_partition_the_scenes(cls, attr, scenes, counted):
    counted(cls, scenes)
    train = getattr(_build(cls, split="train"), attr)
    test = getattr(_build(cls, split="test"), attr)
    assert not set(train) & set(test), "a scene must not be in both splits"
    assert sorted(train + test) == sorted(scenes), "no scene may be dropped"


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_seed_controls_the_split(cls, attr, scenes, counted):
    counted(cls, scenes)
    baseline = getattr(_build(cls), attr)
    assert getattr(_build(cls), attr) == baseline, "same seed must reproduce"
    other = getattr(_build(cls, seed=7), attr)
    assert other != baseline, "a different seed must reorder"
    assert set(other) | set(getattr(_build(cls, seed=7, split="test"), attr)) == set(scenes)


@pytest.mark.parametrize(
    "cls,attr,scenes",
    [s for s in SHARDERS if s[0] is not LandsatIntermediateSharder],
    ids=[n for n in IDS if "Landsat" not in n],
)
def test_max_scenes_caps_before_splitting(cls, attr, scenes, counted):
    """Landsat is excluded: it has no max_scenes parameter. Capping happens
    before the cut, so the train/test ratio is preserved."""
    counted(cls, scenes)
    train = getattr(_build(cls, max_scenes=5), attr)
    assert train == _expected(scenes, max_scenes=5)
    assert len(train) == 4, "int(5 * 0.8)"


@pytest.mark.parametrize("cls,attr,scenes", SHARDERS, ids=IDS)
def test_destination_prefix_encodes_geometry(cls, attr, scenes, counted):
    """build_prefix is the single definition of the shard layout — writer and
    reader both derive from it, so a trainer asking for 128px cannot be handed
    256px shards."""
    counted(cls, scenes)
    sharder = _build(cls, width=128, height=128, stride=64)
    assert sharder.destination_prefix == (
        f"patches/{cls.SENSOR}/train/intermediate/w128_h128_s64/"
    )
