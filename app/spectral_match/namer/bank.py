"""
The family library resampled to one band list, as tensors the generator and the model use (spectral-namer design).

Resampling is the library's own (`build_sensor_cache`, Gaussian SRFs from the band list's FWHM, here 1.1 x band
spacing as in the separability check). Bands a spectrum does not cover are filled by linear interpolation along
wavelength (and the nearest end beyond its range), so every entry has every band. Each entry has a class
(classes.class_of) and a deterministic split: a man-made entry is "test" (never planted in training) when a hash of
its name falls below 0.2 - about 20% overall, 17-30% per class on EnMAP since the hash ignores the class - so every
band list agrees. Repeated names keep their first entry only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.spectral_match.namer.classes import MANMADE_CLASSES, NATURAL, class_of

TEST_SHARE = 0.2


@dataclass
class LibraryBank:
    refl: torch.Tensor          # (L, N) reflectance on the band list
    cls: torch.Tensor           # (L,) class index into MANMADE_CLASSES, -1 for natural
    test: torch.Tensor          # (L,) bool: held out of training
    names: list
    holdout_class: int = -1     # a whole man-made class held out (open-set run), -1 for none

    @property
    def manmade(self) -> torch.Tensor:
        return self.cls >= 0

    def bands(self, keep: torch.Tensor) -> "LibraryBank":
        """The same bank on a subset of the band list (a batch's valid bands)."""
        return LibraryBank(self.refl[:, keep], self.cls, self.test, self.names, self.holdout_class)

    def to(self, device):
        return LibraryBank(self.refl.to(device), self.cls.to(device), self.test.to(device), self.names,
                           self.holdout_class)


def _is_test(name: str) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < TEST_SHARE


def _fill(refl: np.ndarray, wl: np.ndarray) -> np.ndarray:
    out = refl.copy()
    for i, r in enumerate(refl):
        ok = np.isfinite(r)
        if ok.all() or not ok.any():
            continue
        out[i] = np.interp(wl, wl[ok], r[ok])
    return out


def build_bank(wavelengths_nm, cache_dir: str | Path, library_root: str | Path, holdout_class: str | None = None,
               min_coverage: float = 0.9) -> LibraryBank:
    """LibraryBank for one band list (wavelengths in the band list's own order)."""
    from app.spectral_match.library import build_sensor_cache, load_slim_bundle
    wl = np.asarray(wavelengths_nm, dtype=np.float64)
    order = np.argsort(wl)                               # the resampler wants ascending wavelengths
    key = hashlib.sha1(np.round(wl, 2).tobytes()).hexdigest()[:12]
    bundle = load_slim_bundle(library_root)
    entries, _ = build_sensor_cache(bundle=bundle, sensor_id=f"namer_{key}", target_wl_nm=wl[order],
                                    target_fwhm_nm=np.abs(np.gradient(wl[order])) * 1.1, cache_dir=cache_dir,
                                    min_coverage=min_coverage, progress=False)
    keep = [(e, class_of(e.chapter, e.name)) for e in entries]
    keep = [(e, c) for e, c in keep if c is not None and np.isfinite(e.refl).mean() >= min_coverage]
    seen: set = set()                                    # one entry per name: a repeated name ("Copper Metal" x3)
    keep = [(e, c) for e, c in keep if not (e.name in seen or seen.add(e.name))]   # would leak past leave-own-out
    refl = _fill(np.stack([e.refl for e, _ in keep]).astype(np.float64), wl[order])
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    refl = refl[:, inv]                                  # back to the band list's own order
    cls = np.array([-1 if c == NATURAL else MANMADE_CLASSES.index(c) for _, c in keep])
    test = np.array([c != NATURAL and _is_test(e.name) for e, c in keep])
    hold = MANMADE_CLASSES.index(holdout_class) if holdout_class else -1
    return LibraryBank(torch.from_numpy(refl).float(), torch.from_numpy(cls).long(), torch.from_numpy(test),
                       [e.name for e, _ in keep], hold)
