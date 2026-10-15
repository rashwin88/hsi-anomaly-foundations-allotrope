"""
Family tags, pruning and band selection for the family library, feeding family_match.

Called by family_match.match_families on the entries library.load_sensor_cache returns for
the family bundle (data/family_library, built by scripts/curate_family_library.py). That
bundle stores each entry's tag in its chapter as "<group>.<family>", e.g. "manmade.roofing",
so the tag travels through the existing per-sensor cache untouched.

Why families at all: sam.py ranks entries one by one, so ~300 minerals out-vote the man-made
spectra by sheer number of chances. Scoring per family needs every entry labelled and every
family capped (FAMILY_CAP). See docs/lld/spectral-family-match.md.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.spectral_match.library import LibraryEntry

logger = logging.getLogger(__name__)

GROUPS = ("manmade", "natural", "exclude")
# 40 per family is what the experiments used; uncapped, minerals out-vote man-made entries.
FAMILY_CAP = 40


@dataclass(frozen=True)
class TaggedLibrary:
    """Library entries as parallel arrays.

    names (N,) material_id; refl (N, B) float32, NaN where the sensor band is not covered;
    valid (N, B) uint8; group (N,) one of GROUPS; family (N,) str.
    """
    names: np.ndarray
    refl: np.ndarray
    valid: np.ndarray
    group: np.ndarray
    family: np.ndarray

    def subset(self, keep: np.ndarray) -> "TaggedLibrary":
        return TaggedLibrary(self.names[keep], self.refl[keep], self.valid[keep],
                             self.group[keep], self.family[keep])


def parse_tag(chapter: str) -> tuple[str, str] | None:
    """("manmade", "roofing") from "manmade.roofing"; None if not a family tag."""
    group, _, family = chapter.partition(".")
    return (group, family) if group in GROUPS and family else None


def tag_library(entries: Sequence[LibraryEntry]) -> TaggedLibrary:
    """Parallel arrays with each entry's tag from its chapter. An entry without a valid tag
    (e.g. a splib07_slim entry) is excluded, loudly: a silent default would let an untagged
    spectrum change the results unnoticed."""
    tags = [parse_tag(e.chapter) for e in entries]
    missing = [e.material_id for e, t in zip(entries, tags) if t is None]
    if missing:
        logger.warning("%d library entries have no family tag and are excluded, e.g. %s",
                       len(missing), missing[:3])
    gf = [t or ("exclude", "untagged") for t in tags]
    return TaggedLibrary(
        names=np.array([e.material_id for e in entries]),
        refl=np.stack([e.refl for e in entries]).astype(np.float32),
        valid=np.stack([e.valid for e in entries]).astype(np.uint8),
        group=np.array([g for g, _ in gf]),
        family=np.array([f for _, f in gf]),
    )


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def balanced_subset(lib: TaggedLibrary, cap: int = FAMILY_CAP) -> np.ndarray:
    """(N,) bool: excluded entries dropped, at most ``cap`` spectrally spread entries per family.

    Greedy farthest-point by spectral angle, from the entry nearest the family's median
    shape; a family of ``cap`` or fewer is kept whole. Angles use 0 for uncovered bands, so
    run this after common_band_mask has removed entries with real gaps.
    """
    keep = np.zeros(len(lib.names), dtype=bool)
    u = _unit(np.nan_to_num(lib.refl.astype(np.float64)))
    for fam in sorted(set(lib.family[lib.group != "exclude"])):
        idx = np.flatnonzero((lib.family == fam) & (lib.group != "exclude"))
        if idx.size <= cap:
            keep[idx] = True
            continue
        fu = u[idx]
        chosen = [int(np.argmax(fu @ _unit(np.median(fu, axis=0))))]
        dist = np.arccos(np.clip(fu @ fu[chosen[0]], -1, 1))
        while len(chosen) < cap:
            nxt = int(np.argmax(dist))
            chosen.append(nxt)
            dist = np.minimum(dist, np.arccos(np.clip(fu @ fu[nxt], -1, 1)))
        keep[idx[chosen]] = True
    return keep


def common_band_mask(pixel_valid: np.ndarray, lib: TaggedLibrary,
                     min_pixel_share: float = 0.99,
                     min_entry_share: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """(bands (B,) bool, entries (N,) bool): one band set the whitened detectors can share.

    The SAM matcher can intersect validity per pixel/entry pair; whitening and NNLS need a
    single band set. A band is kept if it is valid for >= min_pixel_share of the pixels and
    >= min_entry_share of the non-excluded entries; entries invalid on any kept band are
    then dropped. pixel_valid is (P, B) 1/0 for the pixels that will be scored.
    """
    usable = lib.group != "exclude"
    bands = (pixel_valid.mean(0) >= min_pixel_share) & \
            (lib.valid[usable].mean(0) >= min_entry_share)
    entries = usable & lib.valid[:, bands].all(1)
    n_b, n_e = int((~bands).sum()), int((usable & ~entries).sum())
    if n_b or n_e:
        logger.info("common_band_mask: dropped %d of %d bands and %d of %d entries",
                    n_b, bands.size, n_e, int(usable.sum()))
    return bands, entries
