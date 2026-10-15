"""
Name the material family of anomaly pixels with calibrated Simplex ACE, at 30 m.

Called from the spectral_library_match worker (not wired yet, see the integration recipe in
docs/lld/spectral-family-match.md) with the native onboarding vendable, the anomaly mask
from anomaly_detection_prep, and the family library cache from ensure_family_cache.

Why not whole-pixel SAM (sam.py): a man-made object usually fills a fraction of a 30 m pixel,
so the whole spectrum mostly matches the surroundings. Here each pixel is compared with its
own ring background in whitened space, a man-made family is scored as the cone of all its
spectra (Ziemann & Theiler 2017), and every family score becomes a percentile against the
scene's ordinary pixels. A name is given only if that percentile reaches p* = 1 - alpha / F
(F = man-made families): an ordinary pixel then gets a man-made name with probability at most
alpha. At alpha = 5 % it named 27 % (78 % with field spectra in the library) of the answered
pixels on real deployed sheets correctly, where ACE + feature checks named 0 %.

Inputs must be surface reflectance on the library's 0-1 scale: the scores use pixel minus
background, which, unlike SAM's angle, depends on brightness.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import nnls

from app.spectral_match.family_library import (TaggedLibrary, balanced_subset, common_band_mask,
                                               tag_library)
from app.spectral_match.library import LibraryEntry, build_cache_for_vendable, load_sensor_cache

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05                 # allowed share of ordinary pixels given a man-made name
RING_OUTER, RING_INNER = 13, 5       # ring = 13x13 window minus the central 5x5
MIN_RING_PX = 20                     # fewer usable ring pixels -> fall back to all valid ones
NULL_N = 10_000                      # background pixels for the percentile calibration
EXCLUDE_BUFFER_PX = 3                # anomalies grown by this before sampling the background
WHITEN_SAMPLE_PX = 60_000
RIDGE = 1e-3                         # x mean eigenvalue; the smallest eigenvalues are noise
FAMILY_BUNDLE_DIR = "/srv/family_library"


@dataclass(frozen=True)
class FamilyMatch:
    """Per scored pixel (P,): best family, its group, its calibrated percentile, the closest
    library entry in that family, and whether the percentile reached p_star."""
    family: np.ndarray
    group: np.ndarray
    confidence: np.ndarray
    best_entry: np.ndarray
    answered: np.ndarray
    p_star: float
    n_manmade_families: int
    bands_used: np.ndarray           # (B,) bool over the vendable's bands


def ensure_family_cache(vendable, sensor_id: str, cache_dir: str = "/splib07_cache",
                        bundle_dir: str = FAMILY_BUNDLE_DIR) -> list[LibraryEntry]:
    """The family library resampled to this vendable's bands, built on first use.

    Reuses the splib07 cache machinery; the bundle's own version tag keeps its cache file
    apart from splib07_slim's. chapters=None: every family is wanted.
    """
    diag = build_cache_for_vendable(vendable=vendable, sensor_id=sensor_id,
                                    slim_bundle_dir=bundle_dir, cache_dir=cache_dir,
                                    chapters=None, min_coverage=0.7)
    if "cache_path" not in diag:
        raise RuntimeError(f"family library cache not available: {diag['status']}")
    return load_sensor_cache(diag["cache_path"])


def p_star(alpha: float, n_manmade_families: int) -> float:
    """Percentile bar such that an ordinary pixel gets a man-made name with prob. <= alpha."""
    return 1.0 - alpha / max(n_manmade_families, 1)


def ring_means(cube: np.ndarray, weight: np.ndarray, rows: np.ndarray, cols: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """((N, B) weighted ring mean, (N,) ring weight) at (rows, cols). cube (B, H, W)."""
    def box(a, s):
        return ndimage.uniform_filter(a, size=s, mode="constant") * (s * s)
    w = weight.astype(np.float32)
    den = (box(w, RING_OUTER) - box(w, RING_INNER))[rows, cols]
    out = np.empty((len(rows), cube.shape[0]), np.float64)
    for b in range(cube.shape[0]):
        layer = cube[b] * w
        out[:, b] = (box(layer, RING_OUTER) - box(layer, RING_INNER))[rows, cols]
    return out / np.maximum(den, 1e-9)[:, None], den


def whitener(cube: np.ndarray, valid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """(B, B) symmetric C^(-1/2) of the scene covariance, from a valid-pixel sample."""
    r, c = np.nonzero(valid)
    take = rng.choice(r.size, size=min(WHITEN_SAMPLE_PX, r.size), replace=False)
    lam, V = np.linalg.eigh(np.cov(cube[:, r[take], c[take]].astype(np.float64)))
    lam = np.maximum(lam, 0) + RIDGE * lam.mean()
    return (V / np.sqrt(lam)) @ V.T


def ace(x: np.ndarray, m: np.ndarray, E: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(N, K) signed ACE of pixels x against entries E, both relative to local means m.

    z = W(x - m), t = W(e - m); ACE = sign(t.z)(t.z)^2 / (|t|^2 |z|^2). Matrix products only:
    the (N, K, B) target array would not fit for a full library.
    """
    z, wm, we = (x - m) @ W, m @ W, E @ W
    tz = z @ we.T - (wm * z).sum(1)[:, None]
    tt = (we * we).sum(1)[None, :] - 2 * (wm @ we.T) + (wm * wm).sum(1)[:, None]
    zz = (z * z).sum(1)[:, None]
    return np.sign(tz) * tz * tz / (np.maximum(tt, 1e-12) * np.maximum(zz, 1e-12))


def family_scores(x, m, E, W, fam_cols: list[np.ndarray], fam_is_manmade: np.ndarray) -> np.ndarray:
    """(N, F): Simplex ACE for man-made families, best single-entry ACE for natural ones.

    Simplex ACE = |projection of z on the cone {sum a_k t_k, a >= 0}|^2 / |z|^2 (NNLS). Never
    a cone of natural spectra: dozens of minerals and soils span almost every direction.
    """
    a = ace(x, m, E, W)
    z, we, wm = (x - m) @ W, E @ W, m @ W
    out = np.empty((len(x), len(fam_cols)))
    for j, ix in enumerate(fam_cols):
        if not fam_is_manmade[j]:
            out[:, j] = a[:, ix].max(1)
            continue
        for i in range(len(x)):
            T = we[ix] - wm[i]
            coef, _ = nnls(T.T, z[i], maxiter=500)
            p = T.T @ coef
            out[i, j] = p @ p / max(z[i] @ z[i], 1e-12)
    return out


def calibrate(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Per family, the share of background (null) scores strictly below each pixel's score."""
    out = np.empty_like(scores)
    for j in range(scores.shape[1]):
        out[:, j] = np.searchsorted(np.sort(null[:, j]), scores[:, j], side="left") / len(null)
    return out


def match_families(cube: np.ndarray, validity: np.ndarray, anomaly_mask: np.ndarray,
                   rows: np.ndarray, cols: np.ndarray, entries: list[LibraryEntry],
                   alpha: float = DEFAULT_ALPHA, seed: int = 0) -> FamilyMatch:
    """Family names for the pixels (rows, cols).

    cube (B, H, W) surface reflectance 0-1 on the vendable's bands; validity (B, H, W) 1/0;
    anomaly_mask (H, W) bool, kept out of ring backgrounds and of the calibration sample;
    entries from ensure_family_cache. Raises ValueError if fewer than 2 families survive.
    """
    rng = np.random.default_rng(seed)
    lib: TaggedLibrary = tag_library(entries)
    bands, ent = common_band_mask(validity[:, rows, cols].T, lib)
    lib = lib.subset(ent)
    lib = lib.subset(balanced_subset(lib))
    names = sorted(set(lib.family))
    if len(names) < 2:
        raise ValueError(f"family library left {len(names)} families after band checks")
    fam_cols = [np.flatnonzero(lib.family == f) for f in names]
    fam_mm = np.array([lib.group[ix[0]] == "manmade" for ix in fam_cols])
    E = lib.refl[:, bands].astype(np.float64)
    c = cube[bands].astype(np.float32)
    valid = validity[bands].all(0)
    x = c[:, rows, cols].T.astype(np.float64)
    med = float(np.median(x))
    if not 0.005 < med < 1.5:
        logger.warning("median pixel value %.4g is not 0-1 reflectance; scores will be wrong", med)

    usable = valid & ~anomaly_mask
    m, den = ring_means(c, usable, rows, cols)
    few = den < MIN_RING_PX
    if few.any():
        m[few] = ring_means(c, valid, rows[few], cols[few])[0]
    W = whitener(c, valid, rng)
    scores = family_scores(x, m, E, W, fam_cols, fam_mm)

    grown = ndimage.binary_dilation(anomaly_mask, iterations=EXCLUDE_BUFFER_PX)
    br, bc = np.nonzero(valid & ~grown)
    take = rng.choice(br.size, size=min(NULL_N, br.size), replace=False)
    m_bg, den_bg = ring_means(c, usable, br[take], bc[take])
    ok = den_bg >= MIN_RING_PX
    null = family_scores(c[:, br[take][ok], bc[take][ok]].T.astype(np.float64), m_bg[ok],
                         E, W, fam_cols, fam_mm)

    pct = calibrate(scores, null)
    best = np.argmax(pct + 1e-6 * np.tanh(scores), 1)        # ties at the top: higher raw score
    conf = pct[np.arange(len(best)), best]
    a = ace(x, m, E, W)
    best_entry = np.array([lib.names[fam_cols[f][np.argmax(a[i, fam_cols[f]])]]
                           for i, f in enumerate(best)])
    ps = p_star(alpha, int(fam_mm.sum()))
    return FamilyMatch(family=np.array(names)[best], group=np.where(fam_mm[best], "manmade", "natural"),
                       confidence=conf, best_entry=best_entry, answered=conf >= ps, p_star=ps,
                       n_manmade_families=int(fam_mm.sum()), bands_used=bands)
