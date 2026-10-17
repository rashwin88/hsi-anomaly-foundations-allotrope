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


# --- bench-v2: realistic implants (2026-10-17) ---------------------------------------------------------------------
# Measured on PRISMA (C:/es/implant/v2/diag.py): real objects' difference from their ring lies almost wholly inside
# the scene's top-10 spectral subspace (foreign fraction median 0.006 vs 0.042 for bench-v1 implants) and bleeds into
# the 1-px halo (halo / object contrast 0.74 vs 0.60). v2 matches both.

FOREIGN_K = 10


def background_subspace(cube: np.ndarray, valid: np.ndarray, k: int = FOREIGN_K, seed: int = 0) -> np.ndarray:
    """(k, B) orthonormal rows: the top-k principal directions of the scene's valid pixels. Missing voxels (NaN)
    are mean-imputed per band - a plain mean would make a band with one gap all-NaN and drop it from the subspace
    (130-182 of 234 PRISMA host bands had gaps; found 2026-10-17)."""
    X = cube[:, valid].T
    rng = np.random.default_rng(seed)
    sub = X[rng.choice(len(X), size=min(len(X), 100_000), replace=False)]
    centred = sub - np.nanmean(sub, 0)
    centred[~np.isfinite(centred)] = 0.0                  # gap = the band mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:k]


def foreign_fraction(d: np.ndarray, P: np.ndarray) -> float:
    """Share of |d|^2 outside the subspace spanned by P's rows (NaN bands ignored)."""
    d = np.nan_to_num(d)
    return float(1.0 - np.sum((P @ d) ** 2) / max(np.sum(d ** 2), 1e-18))


def match_foreignness(d: np.ndarray, P: np.ndarray, target: float) -> np.ndarray:
    """d with its out-of-subspace part rescaled so foreign_fraction(result, P) = target (0 <= target < 1):
    d = d_in + beta d_out, beta^2 |d_out|^2 / (|d_in|^2 + beta^2 |d_out|^2) = target."""
    nan = ~np.isfinite(d)
    d0 = np.nan_to_num(d)
    d_in = P.T @ (P @ d0)
    d_out = d0 - d_in
    a, b = np.sum(d_in ** 2), np.sum(d_out ** 2)
    beta = 0.0 if b < 1e-18 else np.sqrt(target / max(1 - target, 1e-9) * a / b)
    out = d_in + beta * d_out
    out[nan] = np.nan
    return out


def psf_weights(obj: np.ndarray, sigma: float) -> np.ndarray:
    """(H, W) implant weights: the footprint blurred by a Gaussian PSF of std sigma px, scaled so the footprint's
    mean weight is 1 (the object keeps its contrast; neighbours get the blur's tail)."""
    if sigma <= 0:
        return obj.astype(np.float64)
    w = ndimage.gaussian_filter(obj.astype(np.float64), sigma, mode="constant")
    return w / max(w[obj].mean(), 1e-12)


FOREIGN_LADDER = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.25, 0.5)   # signal levels tried


def solve_realised(o0, r, sigma, d0, P, z_target, f_target, shrink, tol: float = 1.5):
    """v2 strength and signal for one object, solved on the REALISED object: its difference from the ring is the
    spot's own natural offset (o0 - r) plus shrink * s * d, so a ladder of signal foreignness levels is tried, the
    strength solved for z at each, and the level whose realised foreignness is closest to f_target kept. Returns
    (s, d) or None when the spot is already above z_target or no level comes within a factor tol of f_target
    (its own offset is too foreign). All arrays over the evaluation bands; shrink = 1 - the ring's mean PSF weight."""
    best = None
    for t in FOREIGN_LADDER:
        d = match_foreignness(d0, P, t)
        s, _ = solve_strength(o0, r, sigma, shrink * d, z_target)
        if s is None:                                   # higher levels only get harder: keep the best so far
            break
        gap = abs(np.log(max(foreign_fraction(o0 - r + s * shrink * d, P), 1e-6) / max(f_target, 1e-6)))
        if best is None or gap < best[0]:
            best = (gap, s, d)
    return None if best is None or best[0] > np.log(tol) else (best[1], best[2])


def evaluation_bands(wl: np.ndarray, finite: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(B,) bool: the bands the evaluator reads - product exclusions out, bands mostly empty over valid pixels out."""
    from app.evaluation.datasets import PRODUCT_EXCLUSIONS
    keep = np.array([not any(lo <= w <= hi for lo, hi in PRODUCT_EXCLUSIONS) for w in wl])
    return keep & (finite[:, valid].mean(1) >= 0.6 if valid.any() else keep)


def implant_tile(rng: np.random.Generator, refl: np.ndarray, wl: np.ndarray, real: dict, library: list,
                 n_objects: int, psf_sigma: float = 1.4, sources: tuple = ("signature", "material"),
                 max_size: int = 20, spacing: int = 10) -> np.ndarray:
    """Training-time implants (docs/proposals/detection-supervised-finetuning.md): up to n_objects bench-v2-style
    objects into one tile, in place. refl (B, H, W) float reflectance, NaN = nodata. real: the build script's
    real.json (objects with px, z, foreign, d, scene; wavelengths_nm per scene). library: (B,) spectra on wl, NaN
    where uncovered. Placement anywhere valid (vessels on water are in scope, user 2026-10-17). Returns (H, W) int32
    object ids; an object whose spot cannot reach its targets in 30 tries is skipped."""
    B, H, W = refl.shape
    finite = np.isfinite(refl)
    valid = finite.mean(0) >= 0.9
    gt = np.zeros((H, W), np.int32)
    margin = 7
    eligible = ndimage.binary_erosion(valid, iterations=margin)
    if eligible.sum() < 50 or not real["objects"]:
        return gt
    ev = evaluation_bands(wl, finite, valid)
    P = background_subspace(refl[ev], valid)
    by_tier = {t: [o for o in real["objects"] if tier_of(o["z"]) == t] for t in TIERS}
    blocked = np.zeros((H, W), bool)
    for _ in range(n_objects):
        pool = by_tier[TIERS[int(rng.integers(3))]]
        if not pool:
            continue
        pick = pool[int(rng.integers(len(pool)))]
        size = int(min(real["objects"][int(rng.integers(len(real["objects"])))]["px"], max_size))
        source = sources[int(rng.integers(len(sources)))]
        for _ in range(30):
            free = np.flatnonzero((eligible & ~blocked).ravel())
            if not len(free):
                break
            cy, cx = divmod(int(free[int(rng.integers(len(free)))]), W)
            off = grow_blob(rng, size) + (cy, cx)
            if off.min() < margin or off[:, 0].max() >= H - margin or off[:, 1].max() >= W - margin \
                    or not eligible[off[:, 0], off[:, 1]].all() or blocked[off[:, 0], off[:, 1]].any():
                continue
            y0, y1 = off[:, 0].min() - margin, off[:, 0].max() + margin + 1
            x0, x1 = off[:, 1].min() - margin, off[:, 1].max() + margin + 1
            sub = refl[:, y0:y1, x0:x1]
            obj = np.zeros((y1 - y0, x1 - x0), bool)
            obj[off[:, 0] - y0, off[:, 1] - x0] = True
            ring = ring_mask(obj, valid[y0:y1, x0:x1], exclude=gt[y0:y1, x0:x1] > 0)
            if ring.sum() < 8:
                continue
            rpx = np.where(np.isfinite(sub[:, ring]), sub[:, ring], np.nan)
            r, sigma, o0 = np.nanmean(rpx, 1), np.nanstd(rpx, 1), np.nanmean(sub[:, obj], 1)
            if source == "signature":
                rw = np.asarray(real["wavelengths_nm"][pick["scene"]])
                near = np.abs(wl[:, None] - rw[None]).min(1) <= 15.0
                d0 = np.where(near, np.interp(wl, rw, pick["d"]), np.nan)
            else:
                d0 = library[int(rng.integers(len(library)))] - r
            wgt = psf_weights(obj, psf_sigma)
            got = solve_realised(o0[ev], r[ev], sigma[ev], np.nan_to_num(d0[ev]), P, float(pick["z"]),
                                 float(pick["foreign"]), 1.0 - wgt[ring].mean())
            if got is None:
                continue
            s, d_ev = got
            d = np.where(np.isfinite(d0), d0, 0.0)
            d[ev] = d_ev
            sub += s * wgt[None] * d[:, None, None]
            np.maximum(sub, 0.0, out=sub, where=np.isfinite(sub))
            gt[y0:y1, x0:x1][obj] = gt.max() + 1
            blocked[max(0, y0 - spacing):y1 + spacing, max(0, x0 - spacing):x1 + spacing] = True
            break
    return gt
