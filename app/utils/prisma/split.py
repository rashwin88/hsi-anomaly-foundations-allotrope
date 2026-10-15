"""
Train/test split of PRISMA scenes by site, with the anomaly-benchmark sites held out of both.

Used by scripts/shard_prisma.py once every scene's centre is known (the inventory has no
footprints, so centres come from reading the scenes).

Why by site: PRISMA revisits places; two dates of one site on opposite sides of the split would
leak. Scenes whose centres lie within SITE_KM of each other are one site (single-link). Why hold out
the benchmark sites: the four labelled scenes (app/utils/evaluation/prisma_bench.py) are not in the
inventory, but their places may be, on other dates - training on them would flatter the bench.
Deterministic: a site's side depends only on (seed, its earliest scene id).
"""

import hashlib
import math

SITE_KM = 15.0
BENCH_KM = 30.0
# Centres of the labelled benchmark scenes (lat, lon), read from their geolocation 2026-10-15.
BENCH_SITES = {"set1": (22.458, 69.612), "set2": (23.328, 86.110), "set3+set4": (21.763, 84.015)}


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(min(1.0, h)))


def sites(centres: dict[str, tuple[float, float]], site_km: float = SITE_KM) -> list[list[str]]:
    """Scene ids grouped into sites: connected components of 'centres within site_km'."""
    ids, parent = sorted(centres), {}
    find = lambda i: i if parent.get(i, i) == i else find(parent[i])  # noqa: E731
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if km(centres[a], centres[b]) <= site_km:
                parent[find(b)] = find(a)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values())


def split_scenes(centres: dict[str, tuple[float, float]], test_fraction: float = 0.10, seed: str = "pr-v1",
                 bench_km: float = BENCH_KM) -> dict[str, str]:
    """scene id -> "train" | "test" | "excluded". A site goes to "excluded" if any of its scenes is
    within bench_km of a benchmark site; of the rest, the test_fraction lowest-ranked sites (at least
    one) go to test. Raises ValueError unless 0 < test_fraction < 1."""
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    out, kept = {}, []
    for group in sites(centres):
        if any(km(centres[s], b) <= bench_km for s in group for b in BENCH_SITES.values()):
            out.update({s: "excluded" for s in group})
        else:
            kept.append(group)
    rank = lambda g: int.from_bytes(hashlib.sha256(f"{seed}:{g[0]}".encode()).digest()[:8], "big")  # noqa: E731
    kept.sort(key=rank)
    n_test = max(1, round(test_fraction * len(kept))) if kept else 0
    for i, group in enumerate(kept):
        out.update({s: "test" if i < n_test else "train" for s in group})
    return out
