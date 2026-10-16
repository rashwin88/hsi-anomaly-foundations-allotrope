"""
Implanted-target anomaly benchmark: the pieces that put a synthetic object into a real reflectance scene with a
chosen contrast (docs/lld/implant-bench.md).

Contrast is the quantity the difficulty tiers are cut on, measured exactly as on the real PRISMA bench objects:

    z = mean_b |o_b - r_b| / mean_b sigma_b

where o is the object's mean spectrum, r and sigma the mean and standard deviation of a ring around it (pixels 2-3
steps out, valid, not in another object), and b runs over the bands valid in both. On the 909 real objects z has
median 0.68; its thirds (0.502, 0.934) cut the tiers.

Two ways to implant, both linear in their strength, so a target z is hit by solving for one number:
  - signature (additive): x' = x + a * d, d = a real object's difference from its ring, carried to this scene;
  - material (linear mixing): x' = (1 - f) * x + f * t, t a library spectrum, f the fill fraction (<= 1).
The realised z is measured after implanting and is what the benchmark records - the solved strength is approximate
when the pixels under the object differ from the ring.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

TIER_EDGES = (0.502, 0.934)      # z terciles of the 909 real PRISMA bench objects (2026-10-17)
TIERS = ("subtle", "medium", "easy")


def tier_of(z: float) -> str:
    return TIERS[int(z >= TIER_EDGES[0]) + int(z >= TIER_EDGES[1])]


def ring_mask(obj: np.ndarray, valid: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """(H, W) bool: pixels 2-3 dilation steps from obj, valid, outside `exclude` (other objects)."""
    ring = ndimage.binary_dilation(obj, iterations=3) & ~ndimage.binary_dilation(obj, iterations=1) & valid
    return ring if exclude is None else ring & ~exclude


def contrast_z(cube: np.ndarray, band_valid: np.ndarray, obj: np.ndarray, ring: np.ndarray) -> float:
    """z of obj against ring. cube (B, H, W) reflectance; band_valid (B, H, W) bool. Bands count where every object
    pixel and at least 2 ring pixels are valid."""
    ob, rb = band_valid[:, obj], band_valid[:, ring]               # pixels first: never copy the whole cube
    use = ob.all(1) & (rb.sum(1) >= 2)
    o = cube[:, obj][use].mean(1)
    r = np.where(rb[use], cube[:, ring][use], np.nan)
    return float(np.abs(o - np.nanmean(r, 1)).mean() / (np.nanstd(r, 1).mean() + 1e-6))


def grow_blob(rng: np.random.Generator, size: int) -> np.ndarray:
    """(size, 2) int offsets of a compact 4-connected blob containing (0, 0): grown one pixel at a time from a random
    4-neighbour of the blob, preferring neighbours touching more of it (compact, not stringy)."""
    cells = {(0, 0)}
    while len(cells) < size:
        frontier = {}
        for y, x in cells:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c = (y + dy, x + dx)
                if c not in cells:
                    frontier[c] = frontier.get(c, 0) + 1
        cand = sorted(frontier)
        w = np.array([frontier[c] for c in cand], float) ** 2
        cells.add(cand[rng.choice(len(cand), p=w / w.sum())])
    return np.array(sorted(cells))


def implant_signature(cube: np.ndarray, obj: np.ndarray, d: np.ndarray, a: float) -> None:
    """In place: x' = max(x + a * d, 0) on obj's pixels. d (B,), NaN = band left untouched."""
    add = np.nan_to_num(a * d)[:, None]
    cube[:, obj] = np.maximum(cube[:, obj] + add, 0.0)


def implant_material(cube: np.ndarray, obj: np.ndarray, t: np.ndarray, f: float) -> None:
    """In place: x' = (1 - f) * x + f * t on obj's pixels. t (B,), NaN = band left untouched."""
    ok = np.isfinite(t)
    sub = cube[:, obj]
    sub[ok] = (1 - f) * sub[ok] + f * t[ok, None]
    cube[:, obj] = sub


def solve_strength(o0: np.ndarray, r: np.ndarray, sigma: np.ndarray, step: np.ndarray, z_target: float,
                   s_max: float | None = None, n: int = 400) -> tuple[float | None, float]:
    """Strength s at which the realised contrast z(s) = mean |o0 + s * step - r| / mean sigma reaches z_target.

    o0 is the object's mean spectrum before implanting, so the pixel's own natural difference from its ring counts -
    as it does in the real objects' z. step = d (signature, s = a) or t - o0 (material, s = f, s_max = 1); both
    implants move the object mean along o0 + s * step exactly. z(s) is convex in s; the answer is the first crossing
    at or after its minimum. Returns (s, z(s)); s is None when the spot is already above z_target with nothing
    implanted (choose another spot), and s = s_max with z(s_max) < z_target when the target is out of reach."""
    ok = np.isfinite(step) & np.isfinite(o0) & np.isfinite(r) & np.isfinite(sigma)
    o0, r, step, scale = o0[ok], r[ok], step[ok], sigma[ok].mean() + 1e-6
    z = lambda s: np.abs(o0[None] + np.asarray(s)[:, None] * step[None] - r[None]).mean(1) / scale
    if z([0.0])[0] > z_target:
        return None, float(z([0.0])[0])
    if s_max is None:                                # far enough that z must exceed the target
        s_max = (z_target * scale + np.abs(o0 - r).mean()) / max(np.abs(step).mean(), 1e-9) * 2
    grid = np.linspace(0.0, s_max, n)
    zs = z(grid)
    rising = np.arange(n) >= int(np.argmin(zs))
    above = np.flatnonzero(rising & (zs >= z_target))
    if not len(above):
        return float(s_max), float(zs[-1])
    k = above[0]
    s = grid[k] if k == 0 else grid[k - 1] + (z_target - zs[k - 1]) / (zs[k] - zs[k - 1]) * (grid[k] - grid[k - 1])
    return float(s), float(z([s])[0])
