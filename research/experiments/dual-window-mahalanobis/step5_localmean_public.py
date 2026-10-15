"""
Step 5: does "local mean, global covariance" (step 4's surprise on PRISMA)
also hold on the public benchmarks?

Scores the same inputs as steps 2-3 (MUUFL cube; ABU/San Diego/HYDICE/River
cube; HAD100 cube and Indradhanu residual) with dw.local_mean_scores at
WINDOWS, and writes one json per set next to the earlier results so
compare_auc.py and report.py can merge them.

Gotchas:

  - HAD100's Indradhanu residual is recomputed exactly as in step 3; it was
    not saved there.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "indradhanu-fp-filter"))
import dw_core as dw  # noqa: E402
import step2_muufl as p2  # noqa: E402
import step3_public as p3  # noqa: E402

WINDOWS = [(5, 15), (7, 21), (11, 31)]


def lm(X, keep, gt, src, res, curve):
    res[f"{src} global"] = dw.evaluate((X ** 2).sum(0), keep, gt, curve=curve)
    for (i, o), s in dw.local_mean_scores(X, keep, WINDOWS).items():
        res[f"{src} local-mean {i}/{o}"] = dw.evaluate(s, keep, gt, curve=curve)


def main() -> None:
    out = {}
    for name in p2.SCENES:
        cube, keep, gt = p2.load(name)
        lm(dw.whitened_pca(cube, keep, p2.D), keep, gt, "cube", out.setdefault(name, {}), (0.01, 5))
    (p2.DATA_ROOT / "dual_window_muufl" / "localmean.json").write_text(json.dumps(out, indent=1))

    out = {}
    for name, cube, gt in p3.classic_scenes():
        cube = np.nan_to_num(cube)
        keep = np.abs(cube).sum(0) > 0
        lm(dw.whitened_pca(cube, keep, min(p3.D, cube.shape[0])), keep, gt, "cube", out.setdefault(name, {}), p3.CURVE)
    (p3.OUT / "classic_localmean.json").write_text(json.dumps(out, indent=1))

    import step1_candidates as s1
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
    g = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
    stats = json.loads((s1.REPO / "app" / "constants" / "hyperspectral.json").read_text())
    t_mu, t_sd = np.array(stats["mean"])[:, None, None], np.array(stats["std"])[:, None, None]
    tiles = list(p3.had100_scenes())
    gridded = []
    for _, c, wl, _ in tiles:
        i1 = np.clip(np.searchsorted(wl, g), 1, len(wl) - 1)
        w = ((g - wl[i1 - 1]) / (wl[i1] - wl[i1 - 1]))[:, None, None]
        c = np.nan_to_num(c)
        gridded.append(((1 - w) * c[i1 - 1] + w * c[i1]).astype(np.float32))
    pooled = np.concatenate([x.reshape(len(g), -1) for x in gridded], 1)
    l_mu, l_sd = pooled.mean(1)[:, None, None], pooled.std(1)[:, None, None] + 1e-9
    inferencer = s1.load_indradhanu()
    out = {}
    for (name, cube, wl, gt), x in zip(tiles, gridded):
        res = out.setdefault(name, {})
        keep = np.abs(cube).sum(0) > 0
        good = ~np.any([(wl >= a) & (wl <= b) for a, b in p3.ABSORPTION], 0) & (wl >= 400) & (wl <= 2450)
        lm(dw.whitened_pca(np.nan_to_num(cube[good]), keep, p3.D), keep, gt, "cube", res, p3.CURVE)
        refl = (t_mu + t_sd * (x - l_mu) / l_sd).astype(np.float32)
        H, W = keep.shape
        (t, b_), (l, r) = [((n - s) // 2, n - s - (n - s) // 2) for s, n in
                           ((H, max(128, -(-H // 32) * 32)), (W, max(128, -(-W // 32) * 32)))]
        pad = np.pad(refl, ((0, 0), (t, b_), (l, r)), mode="reflect")
        with torch.no_grad():
            recon = inferencer.predict_full_scene(
                torch.from_numpy(pad), torch.ones(1, *pad.shape[1:])).cpu().numpy()[:, t:t + H, l:l + W]
        lm(dw.whitened_pca(refl - recon, keep, p3.D), keep, gt, "residual", res, p3.CURVE)
    (p3.OUT / "had100_localmean.json").write_text(json.dumps(out, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    logging.disable(logging.INFO)
    main()
