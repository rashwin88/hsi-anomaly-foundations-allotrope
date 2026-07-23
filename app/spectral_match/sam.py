"""Spectral Angle Mapper (SAM) with pattern-bucketed per-pair masking.

For each unknown pixel ``u`` and library entry ``l``,

    theta = arccos( <u, l>_M / (||u||_M ||l||_M) )

where ``M`` is the intersection of the unknown's valid bands and the
library entry's valid bands. Angles are returned in degrees.

The naive form does one tiny (1, |M|) dot product per pair, which is
millions of small operations and runs in minutes. The optimisation we
locked in on Day 2 is:

    1. Group unknown pixels by their valid-band pattern P (a uint8
       bitmask of length B). In real scenes only a handful of distinct
       patterns exist — atmospheric/bbl masks are scene-wide, per-pixel
       saturation adds a few small variants.

    2. For each pattern P:
         - restrict the library to bands in P → (N, |P|)
         - drop library entries whose validity inside P is below
           ``min_coverage`` (these can't compete fairly)
         - zero out remaining NaNs inside both pixel and library matrices;
           since the pixel mask already says "valid for every band in P",
           the only NaNs left are library-internal holes inside P.
         - For library entries that DO have internal holes in P,
           per-pair norms have to be computed on the actual surviving
           bands. We handle this by bucketing the library entries by
           their own sub-pattern within P, then matmul each bucket.

    3. Concatenate the per-pattern top-K results back into the global
       pixel order.

This keeps everything in BLAS-friendly matmuls. For a typical anomaly
scene (a few thousand anomalies, ~1200 library entries, 1-3 dominant
patterns), the whole match runs in <2 seconds on CPU.

Output
------
``match_pixels`` returns three (P, K) arrays:

    angles_deg   float32  — SAM angle per match
    library_ix   int32    — index into the library entries list
    n_bands_used int32    — how many bands the angle was computed on

plus a (P,) mask flagging pixels that fell below the absolute
``min_band_count`` floor — those rows have angles set to NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchResult:
    angles_deg: np.ndarray   # (P, K) float32, NaN where no match
    library_ix: np.ndarray   # (P, K) int32, -1 where no match
    n_bands_used: np.ndarray # (P, K) int32, 0 where no match
    no_match: np.ndarray     # (P,) bool, True where pixel had too few valid bands


def _pack_pattern_key(mask: np.ndarray) -> bytes:
    """Pack a (B,) bool/uint8 mask into a compact bytes key suitable for dict use."""
    return np.packbits(mask.astype(np.uint8, copy=False)).tobytes()


def _safe_l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """L2-normalize each row of (R, C); return (normalized, original-norms)."""
    norms = np.linalg.norm(mat, axis=1)
    safe = np.maximum(norms, eps)
    return mat / safe[:, None], norms


def match_pixels(
    *,
    pixels: np.ndarray,
    pixel_valid: np.ndarray,
    library_refl: np.ndarray,
    library_valid: np.ndarray,
    top_k: int = 5,
    min_coverage: float = 0.7,
    min_band_count: int = 20,
) -> MatchResult:
    """SAM top-K match for a stack of pixels against a library.

    Parameters
    ----------
    pixels : (P, B) float32
        Unknown spectra. NaNs allowed where the pixel is invalid; they
        must also be flagged in ``pixel_valid``.
    pixel_valid : (P, B) uint8
        1 where the pixel band is valid, 0 otherwise.
    library_refl : (N, B) float32
        Resampled library entries. NaNs allowed where SRF coverage was
        insufficient; must match ``library_valid``.
    library_valid : (N, B) uint8
        1/0 mask for library entries.
    top_k : int
    min_coverage : float
        For each pixel pattern P, drop library entries whose validity inside
        P is below this fraction.
    min_band_count : int
        Pixels with fewer than this many valid bands get ``no_match=True``.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    pixel_valid = np.asarray(pixel_valid, dtype=np.uint8)
    library_refl = np.asarray(library_refl, dtype=np.float32)
    library_valid = np.asarray(library_valid, dtype=np.uint8)

    P, B = pixels.shape
    N = library_refl.shape[0]
    if pixel_valid.shape != pixels.shape:
        raise ValueError("pixel_valid shape must match pixels")
    if library_valid.shape != library_refl.shape:
        raise ValueError("library_valid shape must match library_refl")
    if library_refl.shape[1] != B:
        raise ValueError(
            f"band count mismatch: pixels has {B}, library has {library_refl.shape[1]}"
        )
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    K = min(top_k, N)

    # Output buffers, NaN/-1/0 by default.
    angles = np.full((P, K), np.nan, dtype=np.float32)
    lib_ix = np.full((P, K), -1, dtype=np.int32)
    nbands = np.zeros((P, K), dtype=np.int32)
    no_match = np.zeros(P, dtype=bool)

    # Replace NaNs in inputs with zeros so masked dot products work; the
    # validity masks fully describe which entries to use.
    pixels_filled = np.where(pixel_valid.astype(bool), pixels, 0.0)
    lib_filled = np.where(library_valid.astype(bool), library_refl, 0.0)

    # Group pixels by their valid-band pattern.
    pattern_to_rows: dict[bytes, list[int]] = {}
    for r in range(P):
        key = _pack_pattern_key(pixel_valid[r])
        pattern_to_rows.setdefault(key, []).append(r)

    for _pkey, row_ix in pattern_to_rows.items():
        rows = np.asarray(row_ix, dtype=np.int64)
        pat = pixel_valid[rows[0]].astype(bool)  # (B,)
        band_ix = np.where(pat)[0]
        n_p = band_ix.size
        if n_p < min_band_count:
            no_match[rows] = True
            continue

        # Pixel sub-matrix on the pattern's bands: (n_rows, n_p).
        pix_sub = pixels_filled[np.ix_(rows, band_ix)]

        # Library coverage inside this pattern.
        lib_valid_sub = library_valid[:, band_ix]                 # (N, n_p)
        cov = lib_valid_sub.sum(axis=1) / float(n_p)              # (N,)
        keep_lib = np.where(cov >= min_coverage)[0]
        if keep_lib.size == 0:
            no_match[rows] = True
            continue

        lib_sub = lib_filled[np.ix_(keep_lib, band_ix)]           # (n_keep, n_p)
        lib_valid_keep = lib_valid_sub[keep_lib]                  # (n_keep, n_p)

        # Bucket the kept library entries by their own sub-pattern.
        # In practice almost all entries are dense within P, so this yields
        # one big "dense" bucket plus a handful of small ones.
        sub_pattern_to_entries: dict[bytes, list[int]] = {}
        for j in range(keep_lib.size):
            sk = _pack_pattern_key(lib_valid_keep[j])
            sub_pattern_to_entries.setdefault(sk, []).append(j)

        # Pixel L2 norms on the full pattern bands (will need adjustment
        # for buckets where library has internal holes inside P).
        pix_norms_full = np.linalg.norm(pix_sub, axis=1)          # (n_rows,)

        # Accumulators for this pattern's pixels.
        # (n_rows, n_keep) — store cosines, then pick top-K across all
        # buckets together.
        cos_full = np.full((rows.size, keep_lib.size), -np.inf, dtype=np.float32)
        nbands_full = np.zeros((rows.size, keep_lib.size), dtype=np.int32)

        for _sk, sub_entries in sub_pattern_to_entries.items():
            sub_ix = np.asarray(sub_entries, dtype=np.int64)
            entry_mask = lib_valid_keep[sub_ix[0]].astype(bool)    # (n_p,)
            n_m = int(entry_mask.sum())
            if n_m < min_band_count:
                continue

            band_sub_ix = np.where(entry_mask)[0]
            # Pixel block restricted to bucket-valid bands: (n_rows, n_m)
            pix_bucket = pix_sub[:, band_sub_ix]
            lib_bucket = lib_sub[sub_ix][:, band_sub_ix]           # (n_sub, n_m)

            pix_norms = np.linalg.norm(pix_bucket, axis=1)
            lib_norms = np.linalg.norm(lib_bucket, axis=1)
            denom = np.outer(np.maximum(pix_norms, 1e-12),
                             np.maximum(lib_norms, 1e-12))         # (n_rows, n_sub)
            dot = pix_bucket @ lib_bucket.T                        # (n_rows, n_sub)
            cos_bucket = np.clip(dot / denom, -1.0, 1.0).astype(np.float32)

            cos_full[:, sub_ix] = cos_bucket
            nbands_full[:, sub_ix] = n_m

        # Top-K per row by largest cosine. -np.inf entries (no valid bucket)
        # naturally fall to the bottom.
        # argpartition is O(n_keep) per row; argsort would be O(n_keep log n_keep).
        k_eff = min(K, keep_lib.size)
        if k_eff == keep_lib.size:
            order = np.argsort(-cos_full, axis=1)
        else:
            part = np.argpartition(-cos_full, k_eff - 1, axis=1)[:, :k_eff]
            # Sort within those k for stable ranking.
            part_cos = np.take_along_axis(cos_full, part, axis=1)
            sort_within = np.argsort(-part_cos, axis=1)
            order = np.take_along_axis(part, sort_within, axis=1)
        topk_ix = order[:, :k_eff]                                 # local indices into keep_lib

        topk_cos = np.take_along_axis(cos_full, topk_ix, axis=1)
        topk_nb = np.take_along_axis(nbands_full, topk_ix, axis=1)

        # Pixels whose top-1 is -inf had no usable bucket — flag no_match.
        bad_rows = ~np.isfinite(topk_cos[:, 0])
        if bad_rows.any():
            no_match[rows[bad_rows]] = True

        good = ~bad_rows
        if good.any():
            angles_block = np.rad2deg(np.arccos(topk_cos[good])).astype(np.float32)
            lib_ix_block = keep_lib[topk_ix[good]].astype(np.int32)
            nb_block = topk_nb[good].astype(np.int32)
            tgt_rows = rows[good]
            angles[tgt_rows, :k_eff] = angles_block
            lib_ix[tgt_rows, :k_eff] = lib_ix_block
            nbands[tgt_rows, :k_eff] = nb_block

    # Silence the pix_norms_full lint — kept as documentation of the
    # full-pattern norm even though buckets use their own bands.
    del pix_norms_full

    return MatchResult(
        angles_deg=angles,
        library_ix=lib_ix,
        n_bands_used=nbands,
        no_match=no_match,
    )
