"""
Step 1: dual-window Mahalanobis on the four annotated PRISMA scenes.

Two inputs are scored, each as 30 whitened principal components:

  - Indradhanu residual (cube − reconstruction): last week's winner, scored
    globally, now scored against a local ring.
  - The cube itself: classical local RX (LRX), the non-model baseline.

For each, every (inner, outer) window in WINDOWS is scored, alongside the
global score (whitened ‖x‖², which reproduces step1b's residual-Mahalanobis
and PCA30-RX) and Indradhanu L1 from indradhanu-fp-filter/step1.

Run as `python -X utf8 step1_prisma.py [cache|sweep]`; `cache` runs the
model once per scene and saves the PCA stacks (~180 MB each) under OUT_ROOT,
`sweep` reads them and writes results.json. Both by default.

Gotchas:

  - The window choice made BEFORE looking at results is PRIOR_PICK; the
    rationale is in README.md. The sweep is reported against it, plus a
    leave-one-scene-out pick, so the headline is not tuned on its own test.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "indradhanu-fp-filter"))
import dw_core as dw  # noqa: E402
import step1_candidates as s1  # noqa: E402

OUT_ROOT = s1.SCENES_ROOT / "dual_window"
D = 30
INNERS = (3, 5, 7, 9, 11, 15, 21)
OUTERS = (15, 21, 31, 41, 51, 61)
WINDOWS = [(i, o) for i in INNERS for o in OUTERS if o >= i + 6]
PRIOR_PICK = (11, 31)


def cache() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    inferencer = s1.load_indradhanu()
    for n, (he5, _) in enumerate(s1.find_scenes(), start=1):
        cube, keep = s1.build_vendable_cube(he5)
        with torch.no_grad():
            recon = inferencer.predict_full_scene(
                torch.from_numpy(cube), torch.from_numpy(keep[None].astype(np.float32))
            ).cpu().numpy()
        np.save(OUT_ROOT / f"set{n}_residual_pca{D}.npy", dw.whitened_pca(cube - recon, keep, D))
        np.save(OUT_ROOT / f"set{n}_cube_pca{D}.npy", dw.whitened_pca(cube, keep, D))
        print(f"set{n} cached", flush=True)
        del cube, recon


def sweep() -> None:
    results = {}
    for n in range(1, 5):
        d1 = s1.OUT_ROOT / f"set{n}"
        keep, gt = np.load(d1 / "keep.npy"), np.load(d1 / "gt.npy")
        res = results[f"set{n}"] = {"Indradhanu L1": dw.evaluate(np.load(d1 / "score_L1.npy"), keep, gt)}
        for src in ("residual", "cube"):
            X = np.load(OUT_ROOT / f"set{n}_{src}_pca{D}.npy")
            res[f"{src} global"] = dw.evaluate((X ** 2).sum(0), keep, gt)
            t0 = time.time()
            scores, fb = dw.dual_window_scores(X, keep, WINDOWS)
            for (i, o), s in scores.items():
                res[f"{src} dw {i}/{o}"] = dw.evaluate(s, keep, gt) | {"fallback": fb[(i, o)]}
            if src == "residual":
                np.save(OUT_ROOT / f"set{n}_residual_dw_{PRIOR_PICK[0]}_{PRIOR_PICK[1]}.npy",
                        scores[PRIOR_PICK])
            print(f"set{n} {src}: {len(WINDOWS)} windows in {time.time() - t0:.0f}s", flush=True)
    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    logging.disable(logging.INFO)
    stages = sys.argv[1:] or ["cache", "sweep"]
    if "cache" in stages:
        cache()
    if "sweep" in stages:
        sweep()
