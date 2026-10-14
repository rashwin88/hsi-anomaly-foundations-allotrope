"""
Train/test split of EnMAP scenes by datatake, stratified on cloud.

Used by scripts/shard_enmap.py before any tile is cut, so a scene's tiles all land on one side.

Why by datatake, not by scene: an EnMAP datatake is one continuous pass, cut into consecutive
tiles (`..._001`, `..._002`, ...) that border each other. Splitting by scene would put a place's
neighbouring stretch in test while its other half trains. Why stratified: cloud is rare (37 of 212
scenes screened on 2026-08-25 had any), so a plain random split can leave a test set with no
cloud at all - useless for the cloud/water head these shards are meant to feed.

Deterministic: a datatake's side depends only on (seed, datatake id), like the SpectralEarth split.
"""

import hashlib
from collections import defaultdict

# A datatake counts as cloudy when any of its scenes reports more cloud than this, in percent
# of the scene (METADATA.XML qualityFlag/cloudCover).
CLOUDY_PCT = 5.0


def _rank(seed: str, datatake: str) -> float:
    digest = hashlib.sha256(f"{seed}:{datatake}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def split_datatakes(scenes: list[dict], test_fraction: float = 0.10, seed: str = "en-v1",
                    cloudy_pct: float = CLOUDY_PCT) -> set[str]:
    """Datatake ids for the test side. `scenes` are dicts with `datatake` and
    `cover_pct["cloud"]` (EnmapScene.facts). Within each stratum (cloudy / clear datatakes) the
    `test_fraction` lowest-ranked datatakes go to test - at least one per non-empty stratum.
    Raises ValueError unless 0 < test_fraction < 1."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    cloud: dict[str, float] = defaultdict(float)
    for s in scenes:
        cloud[s["datatake"]] = max(cloud[s["datatake"]], float(s["cover_pct"]["cloud"]))
    test = set()
    for cloudy in (True, False):
        stratum = sorted((d for d, c in cloud.items() if (c > cloudy_pct) == cloudy), key=lambda d: _rank(seed, d))
        if stratum:
            test.update(stratum[: max(1, round(test_fraction * len(stratum)))])
    return test
