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
SCORE_CHUNK_PX = 20_000              # pixels scored at once; bounds memory on AVIRIS-NG scenes
FAMILY_BUNDLE_DIR = "/srv/family_library"


@dataclass(frozen=True)
class FamilyMatch:
    """Per requested pixel (P,): best family, its group, its calibrated percentile, the closest
    library entry in that family, and whether the percentile reached p_star.

    A pixel invalid on any of bands_used is not scored: its cube values there are nodata fill
    (-9999 on AVIRIS-NG, the scale minimum on PRISMA), not reflectance. It gets scored False,
    family "", group "none", confidence 0 and answered False."""
    family: np.ndarray
    group: np.ndarray
    confidence: np.ndarray
    best_entry: np.ndarray
    answered: np.ndarray
    scored: np.ndarray
    p_star: float
    n_manmade_families: int
    bands_used: np.ndarray           # (B,) bool over the vendable's bands
    family_names: tuple[str, ...]    # every family that survived the band checks, sorted
    family_groups: tuple[str, ...]   # "manmade" | "natural", parallel to family_names


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


def ring_means(cube: np.ndarray, bands: np.ndarray, weight: np.ndarray, rows: np.ndarray,
               cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """((N, len(bands)) float32 weighted ring mean, (N,) ring weight) at (rows, cols).

    cube (B, H, W), possibly a memmap; bands are indices into its first axis. One band is in
    memory at a time: an AVIRIS-NG scene is ~4 GB across its kept bands.
    """
    def box(a, s):
        return ndimage.uniform_filter(a, size=s, mode="constant") * (s * s)
    w = weight.astype(np.float32)
    den = (box(w, RING_OUTER) - box(w, RING_INNER))[rows, cols]
    out = np.empty((len(rows), len(bands)), np.float32)
    for j, b in enumerate(bands):
        layer = np.asarray(cube[b], dtype=np.float32) * w
        out[:, j] = (box(layer, RING_OUTER) - box(layer, RING_INNER))[rows, cols]
    return out / np.maximum(den, 1e-9)[:, None], den


def spectra(cube: np.ndarray, bands: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(N, len(bands)) float64 pixel spectra at (rows, cols) on the given band indices."""
    return np.asarray(cube[:, rows, cols])[bands].T.astype(np.float64)


def whitener(cube: np.ndarray, bands: np.ndarray, valid: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """(B', B') symmetric C^(-1/2) of the scene covariance on `bands`, from a valid-pixel sample."""
    r, c = np.nonzero(valid)
    take = rng.choice(r.size, size=min(WHITEN_SAMPLE_PX, r.size), replace=False)
    lam, V = np.linalg.eigh(np.cov(spectra(cube, bands, r[take], c[take]).T))
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
    bix = np.flatnonzero(bands)
    valid = np.ones(validity.shape[1:], dtype=bool)
    for b in bix:                                    # band by band: validity may be a memmap
        valid &= np.asarray(validity[b], dtype=bool)
    ps = p_star(alpha, int(fam_mm.sum()))
    n_req = len(rows)
    scored = valid[rows, cols]
    out = FamilyMatch(family=np.full(n_req, "", dtype=object), group=np.full(n_req, "none", dtype=object),
                      confidence=np.zeros(n_req), best_entry=np.full(n_req, "", dtype=object),
                      answered=np.zeros(n_req, dtype=bool), scored=scored, p_star=ps,
                      n_manmade_families=int(fam_mm.sum()), bands_used=bands, family_names=tuple(names),
                      family_groups=tuple("manmade" if mm else "natural" for mm in fam_mm))
    if not scored.any():
        return out
    rows, cols = rows[scored], cols[scored]
    med = float(np.median(spectra(cube, bix, rows[:SCORE_CHUNK_PX], cols[:SCORE_CHUNK_PX])))
    if not 0.005 < med < 1.5:
        logger.warning("median pixel value %.4g is not 0-1 reflectance; scores will be wrong", med)

    # Ring means for the scored pixels and the calibration sample in one pass over the bands.
    usable = valid & ~anomaly_mask
    grown = ndimage.binary_dilation(anomaly_mask, iterations=EXCLUDE_BUFFER_PX)
    br, bc = np.nonzero(valid & ~grown)
    take = rng.choice(br.size, size=min(NULL_N, br.size), replace=False)
    br, bc = br[take], bc[take]
    n = len(rows)
    m_all, den_all = ring_means(cube, bix, usable, np.r_[rows, br], np.r_[cols, bc])
    m, den, m_bg, den_bg = m_all[:n], den_all[:n], m_all[n:], den_all[n:]
    few = den < MIN_RING_PX
    if few.any():
        m[few] = ring_means(cube, bix, valid, rows[few], cols[few])[0]
    W = whitener(cube, bix, valid, rng)

    # Scored in chunks so memory stays flat: (pixels x bands) and (pixels x entries) float64
    # arrays for 200k AVIRIS-NG pixels would be several GB each.
    scores = np.empty((n, len(fam_cols)))
    nearest = np.empty((n, len(fam_cols)), dtype=np.int64)     # best single entry per family
    for s in range(0, n, SCORE_CHUNK_PX):
        sl = slice(s, s + SCORE_CHUNK_PX)
        x, mc = spectra(cube, bix, rows[sl], cols[sl]), m[sl].astype(np.float64)
        scores[sl] = family_scores(x, mc, E, W, fam_cols, fam_mm)
        a = ace(x, mc, E, W)
        nearest[sl] = np.stack([ix[np.argmax(a[:, ix], 1)] for ix in fam_cols], 1)
    ok = den_bg >= MIN_RING_PX
    null = family_scores(spectra(cube, bix, br[ok], bc[ok]), m_bg[ok].astype(np.float64),
                         E, W, fam_cols, fam_mm)

    pct = calibrate(scores, null)
    best = np.argmax(pct + 1e-6 * np.tanh(scores), 1)        # ties at the top: higher raw score
    conf = pct[np.arange(len(best)), best]
    best_entry = lib.names[nearest[np.arange(n), best]]
    out.family[scored] = np.array(names)[best]
    out.group[scored] = np.where(fam_mm[best], "manmade", "natural")
    out.confidence[scored] = conf
    out.best_entry[scored] = best_entry
    out.answered[scored] = conf >= ps
    return out
