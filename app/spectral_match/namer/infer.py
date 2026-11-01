"""
Naming detected pixels with a trained spectral namer (docs/proposals/spectral-namer.md).

    namer = load_namer(checkpoint)                    # a train.py checkpoint (last.pt / best.pt / ep<n>.pt)
    out = name_pixels(namer, cube, wavelengths, pixels, library_root, cache_dir)

For each requested pixel: P(man-made), the best man-made class, P(class | man-made) and the fill estimate. The ring
mean (3-7 px, valid pixels only) is computed band by band so a full scene never needs a second cube in memory. The
library side is the whole family library at the scene's bands - test-split spectra included, since at inference
every spectrum is fair to match against. Bands invalid at more than half the requested pixels are dropped, as in
training. The cube is never copied whole (a Tanager or AVIRIS-NG scene is several GB): validity and finiteness are
read band by band, and only the requested pixels' spectra are gathered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.spectral_match.namer.bank import build_bank
from app.spectral_match.namer.classes import MANMADE_CLASSES
from app.spectral_match.namer.generator import INNER, OUTER
from app.spectral_match.namer.model import SpectralNamer


@dataclass
class NamerResult:
    p_manmade: np.ndarray       # (P,) float
    cls: np.ndarray             # (P,) int, index into classes
    p_cls: np.ndarray           # (P,) float, P(class | man-made) of that class
    fill: np.ndarray            # (P,) float, estimated fill fraction
    classes: tuple
    probs: np.ndarray | None = None   # (P, C) float, P(class | man-made) of every class (objects average these)


def load_namer(checkpoint: str | Path, device: str = "cpu") -> SpectralNamer:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["model"]
    model = SpectralNamer(sd["stats.wl"].tolist(), sd["stats.mean"].tolist(), sd["stats.std"].tolist(),
                          **(ck.get("config", {}).get("model") or {}))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected or set(missing) - {"l1_bias"}:                 # l1_bias: checkpoints from before level-1 matching
        raise ValueError(f"namer checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    model.level1 = (ck.get("config") or {}).get("level1", "head")
    return model.to(device).eval()


def ring_means(cube: np.ndarray, valid: np.ndarray | None, rows: np.ndarray, cols: np.ndarray,
               bands: np.ndarray | None = None) -> np.ndarray:
    """(P, len(bands)) mean of valid, finite pixels 3-7 px from each (row, col) - box sums of OUTER minus INNER, one
    band at a time. bands: indices into the cube's first axis (default all); valid None means finite only."""
    from scipy.ndimage import uniform_filter

    bands = np.arange(cube.shape[0]) if bands is None else np.asarray(bands)
    out = np.zeros((len(rows), len(bands)), np.float32)
    for j, b in enumerate(bands):
        ok = np.isfinite(cube[b]) if valid is None else np.asarray(valid[b], bool) & np.isfinite(cube[b])
        m = ok.astype(np.float32)
        v = np.where(ok, cube[b], 0).astype(np.float32)
        s = uniform_filter(v, OUTER, mode="constant") * OUTER ** 2 - uniform_filter(v, INNER, mode="constant") * INNER ** 2
        c = uniform_filter(m, OUTER, mode="constant") * OUTER ** 2 - uniform_filter(m, INNER, mode="constant") * INNER ** 2
        out[:, j] = s[rows, cols] / np.maximum(c[rows, cols], 1)
    return out


@torch.no_grad()
def name_pixels(model: SpectralNamer, cube: np.ndarray, wavelengths, pixels: np.ndarray, library_root: str | Path,
                cache_dir: str | Path, valid: np.ndarray | None = None, batch: int = 4096) -> NamerResult:
    """cube (N, H, W) reflectance; pixels (H, W) bool - the pixels to name (e.g. a committed anomaly mask)."""
    wl = np.asarray(wavelengths, dtype=np.float64)
    rows, cols = np.nonzero(pixels)
    C = len(MANMADE_CLASSES)
    if not len(rows):
        e = np.zeros(0)
        return NamerResult(e, e.astype(int), e, e, MANMADE_CLASSES, np.zeros((0, C)))
    spec = np.asarray(cube[:, rows, cols], dtype=np.float32)                     # (N, P), the requested pixels only
    pv = np.isfinite(spec) if valid is None else np.asarray(valid[:, rows, cols], bool) & np.isfinite(spec)
    bix = np.flatnonzero(pv.mean(axis=1) > 0.5)
    wl = wl[bix]
    device = next(model.parameters()).device
    x = torch.from_numpy(np.nan_to_num(spec[bix].T))
    r = torch.from_numpy(ring_means(cube, valid, rows, cols, bands=bix))
    bv = torch.from_numpy(np.ascontiguousarray(pv[bix].T))
    w = torch.from_numpy(wl).float().to(device)
    bank = build_bank(wl, cache_dir, library_root).to(device)
    lib = model.forward_library(bank.refl, w)
    matching = getattr(model, "level1", "head") == "matching"     # level 1 by matching: natural entries as class C
    lib_cls = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, C if matching else -1))
    pm, cl, pc, fl, pp = [], [], [], [], []
    for i in range(0, len(x), batch):
        z, logit, fill = model.forward_pixel(x[i:i + batch].to(device), r[i:i + batch].to(device), w,
                                             bv[i:i + batch].to(device))
        scores = model.class_scores(z, lib, lib_cls, C + 1 if matching else C)
        if matching:
            logit, scores = model.level1_logit(scores), scores[:, :C]
        p = scores.softmax(1)
        best = p.max(1)
        pm.append(torch.sigmoid(logit)); cl.append(best.indices); pc.append(best.values); fl.append(fill); pp.append(p)
    cat = lambda t: torch.cat(t).cpu().numpy()
    return NamerResult(cat(pm), cat(cl), cat(pc), cat(fl), MANMADE_CLASSES, cat(pp))
