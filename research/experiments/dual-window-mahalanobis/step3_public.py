"""
Step 3: the dual-window sweep on the public anomaly benchmarks Vinoth pulled
from the shared Drive folder (unzipped into BENCH).

Two result files:

  - classic.json: ABU airport/beach/urban (13), San Diego, HYDICE Urban and
    WHU-Hi-River. These carry no wavelengths, so only the cube is scored.
  - had100.json: the 94 HAD100 AVIRIS-NG test tiles. These have wavelengths,
    so Indradhanu runs too, and its residual is scored like PRISMA's.

Gotchas:

  - HAD100 is at-sensor radiance and Indradhanu was trained on reflectance.
    Each band is mapped affinely onto the model's training mean/std
    (app/constants/hyperspectral.json), using ONE fit pooled over all 94
    tiles. That is a crude empirical line: it gets the scale and spectral
    shape roughly right, but it does not remove the atmosphere.
  - HAD100 tiles are 64×64 or 100×100 and the model's patch is 128, so each
    tile is mirror-padded to at least 128 (a multiple of 32) and cropped back.
  - AVIRIS-NG here is about 2-5 m per pixel, against the ~30 m Indradhanu
    trained on, so the model also sees textures it never learned.
  - The window grid is smaller than PRISMA's because the scenes are only
    64-150 px across. PRIOR_PICK was fixed from the ground-truth sizes
    before any result was seen (see README).
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "indradhanu-fp-filter"))
import dw_core as dw  # noqa: E402

BENCH = Path(r"C:\Users\FITT\Documents\public_had_datasets\bench")
OUT = BENCH.parent / "dual_window_public"
D = 30
INNERS = (3, 5, 7, 9, 11)
OUTERS = (9, 11, 15, 21, 31)
WINDOWS = [(i, o) for i in INNERS for o in OUTERS if o >= i + 4]
PRIOR_PICK = (9, 21)
CURVE = (0.05, 10)           # top-% range for the false-alarm curve on small scenes
ABSORPTION = ((1340, 1460), (1790, 1960))
ENVI_DTYPES = {1: np.uint8, 2: np.int16, 4: np.float32, 5: np.float64, 12: np.uint16}


def read_envi(hdr: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(bands, H, W) array and wavelengths (or None) from an ENVI header + its data file."""
    text = hdr.read_text(errors="ignore")
    get = lambda k: re.search(rf"^{k}\s*=\s*(.+)$", text, re.M | re.I).group(1).strip()  # noqa: E731
    W, H, B = int(get("samples")), int(get("lines")), int(get("bands"))
    dt = np.dtype(ENVI_DTYPES[int(get("data type"))]).newbyteorder(">" if get("byte order") == "1" else "<")
    data = next(p for p in hdr.parent.glob(hdr.stem + ".*") if p.suffix.lower() in (".dat", ".img", ""))
    raw = np.fromfile(data, dtype=dt, count=W * H * B)
    order = {"bsq": (B, H, W), "bil": (H, B, W), "bip": (H, W, B)}[get("interleave").lower()]
    cube = raw.reshape(order)
    cube = {"bsq": cube, "bil": cube.transpose(1, 0, 2), "bip": cube.transpose(2, 0, 1)}[get("interleave").lower()]
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.I)
    wl = np.array([float(v) for v in m.group(1).split(",")]) if m else None
    return cube.astype(np.float32), wl


def classic_scenes():
    """(name, (bands, H, W) cube, gt) for the scenes without wavelengths."""
    for f in sorted(BENCH.glob("abu-*.mat")) + [BENCH / "Sandiego.mat", BENCH / "HYDICE-urban.mat"]:
        m = sio.loadmat(f)
        yield f.stem, np.moveaxis(m["data"].astype(np.float32), -1, 0), m["map"] > 0
    cube, _ = read_envi(BENCH / "whu_river" / "WHU-Hi-River.hdr")
    gt, _ = read_envi(BENCH / "whu_river" / "Anomaly detection" / "Target_mask.hdr")
    yield "WHU-Hi-River", cube, gt[0] > 0


def had100_scenes():
    """(name, radiance cube, wavelengths, gt) for every HAD100 AVIRIS-NG test tile."""
    for hdr in sorted((BENCH / "HAD100" / "data" / "aviris_ng_target").glob("*.hdr")):
        cube, wl = read_envi(hdr)
        gt = sio.loadmat(BENCH / "HAD100" / "gt" / "aviris_ng_gt" / f"{hdr.stem}.mat")["gt"] > 0
        yield hdr.stem, cube, wl, gt


def score_all(X: np.ndarray, keep: np.ndarray, gt: np.ndarray, src: str, res: dict) -> None:
    """Global and every dual-window score of one whitened stack, evaluated into res."""
    res[f"{src} global"] = dw.evaluate((X ** 2).sum(0), keep, gt, curve=CURVE)
    scores, fb = dw.dual_window_scores(X, keep, WINDOWS)
    for (i, o), s in scores.items():
        res[f"{src} dw {i}/{o}"] = dw.evaluate(s, keep, gt, curve=CURVE) | {"fallback": fb[(i, o)]}


def run_classic() -> None:
    results = {}
    for name, cube, gt in classic_scenes():
        cube = np.nan_to_num(cube)
        keep = np.abs(cube).sum(0) > 0
        X = dw.whitened_pca(cube, keep, min(D, cube.shape[0]))
        score_all(X, keep, gt, "cube", results.setdefault(name, {}))
        print(name, cube.shape, int(gt.sum()), "gt px", flush=True)
    (OUT / "classic.json").write_text(json.dumps(results, indent=1))


def run_had100() -> None:
    import step1_candidates as s1
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
    from app.utils.anomaly_detection.scoring import compute_score

    grid = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
    stats = json.loads((s1.REPO / "app" / "constants" / "hyperspectral.json").read_text())
    t_mu, t_sd = np.array(stats["mean"])[:, None, None], np.array(stats["std"])[:, None, None]
    tiles = list(had100_scenes())

    def to_grid(cube, wl):
        """Linear interpolation of every pixel onto the 165-band grid (wl ascending)."""
        i1 = np.clip(np.searchsorted(wl, grid), 1, len(wl) - 1)
        w = ((grid - wl[i1 - 1]) / (wl[i1] - wl[i1 - 1]))[:, None, None]
        return ((1 - w) * cube[i1 - 1] + w * cube[i1]).astype(np.float32)

    gridded = [to_grid(np.nan_to_num(c), wl) for _, c, wl, _ in tiles]
    pooled = np.concatenate([g.reshape(len(grid), -1) for g in gridded], 1)
    l_mu, l_sd = pooled.mean(1)[:, None, None], pooled.std(1)[:, None, None] + 1e-9

    inferencer = s1.load_indradhanu()
    results = {}
    for (name, cube, wl, gt), g in zip(tiles, gridded):
        res = results[name] = {}
        keep = np.abs(cube).sum(0) > 0
        good = ~np.any([(wl >= a) & (wl <= b) for a, b in ABSORPTION], 0) & (wl >= 400) & (wl <= 2450)
        score_all(dw.whitened_pca(np.nan_to_num(cube[good]), keep, D), keep, gt, "cube", res)

        refl = (t_mu + t_sd * (g - l_mu) / l_sd).astype(np.float32)
        H, W = keep.shape
        (t, b_), (l, r) = [((n - s) // 2, n - s - (n - s) // 2) for s, n in
                           ((H, max(128, -(-H // 32) * 32)), (W, max(128, -(-W // 32) * 32)))]
        pad = np.pad(refl, ((0, 0), (t, b_), (l, r)), mode="reflect")
        with torch.no_grad():
            recon = inferencer.predict_full_scene(
                torch.from_numpy(pad), torch.ones(1, *pad.shape[1:])).cpu().numpy()[:, t:t + H, l:l + W]
        res["Indradhanu L1"] = dw.evaluate(compute_score(refl, recon, keep, method="L1"), keep, gt, curve=CURVE)
        score_all(dw.whitened_pca(refl - recon, keep, D), keep, gt, "residual", res)
        print(name, flush=True)
    (OUT / "had100.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    logging.disable(logging.INFO)
    OUT.mkdir(exist_ok=True)
    stages = sys.argv[1:] or ["classic", "had100"]
    if "classic" in stages:
        run_classic()
    if "had100" in stages:
        run_had100()
