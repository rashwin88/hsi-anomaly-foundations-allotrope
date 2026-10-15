"""
Step 2: the same dual-window sweep on the public MUUFL Gulfport campus scenes.

MUUFL is the one target-detection set in the shared Drive folder with real
ground truth: 64 cloth panels per flight (0.5, 1, 3 and 6 m, ~1 m pixels), at
surveyed row/col positions. The Drive copy's ready-made masks are empty, so the
truth is read from the public GatorSense repository (git clone into
DATA_ROOT/MUUFLGulfport).

Only the cube is scored, not an Indradhanu residual: MUUFL covers 368-1043 nm
and Indradhanu needs the 460-2450 nm common grid, so most of its input bands
would be invented. This run tests the window-size rule, not the model.

Gotchas:

  - Truth positions are MATLAB 1-based; converted here.
  - A panel is drawn as a square of ceil(size / 1 m) px on its surveyed centre;
    0.5 and 1 m panels are a single, mostly sub-pixel, pixel. Several are
    under tree canopy (Targets_HumanConf), and they are kept.
  - The evaluation uses a 2 px neutral zone, like PRISMA, to absorb the
    geolocation error in the surveyed positions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dw_core as dw  # noqa: E402

DATA_ROOT = Path(r"C:\Users\FITT\Documents\public_had_datasets")
SCENES = ("campus_w_lidar_1", "campus_3", "campus_4")
D = 30
INNERS = (3, 5, 7, 9, 11, 15)
OUTERS = (11, 15, 21, 31, 41)
WINDOWS = [(i, o) for i in INNERS for o in OUTERS if o >= i + 6]


def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(bands, H, W) cube, keep mask and target mask for one campus scene."""
    h = sio.loadmat(DATA_ROOT / "MUUFLGulfport" / "MUUFLGulfportDataCollection" / f"muufl_gulfport_{name}.mat",
                    squeeze_me=True, struct_as_record=False)["hsi"]
    cube = np.moveaxis(h.Data.astype(np.float32), -1, 0)
    keep = np.isfinite(cube).all(0) & (np.abs(cube).sum(0) > 0)
    gt = np.zeros(keep.shape, bool)
    t = h.groundTruth
    for r, c, s in zip(np.atleast_1d(t.Targets_rowIndices), np.atleast_1d(t.Targets_colIndices),
                       np.atleast_1d(t.Targets_Size)):
        k = max(1, int(np.ceil(s)))
        r0, c0 = int(r) - 1 - (k - 1) // 2, int(c) - 1 - (k - 1) // 2
        gt[max(0, r0):r0 + k, max(0, c0):c0 + k] = True
    return np.nan_to_num(cube), keep, gt


def main() -> None:
    out = DATA_ROOT / "dual_window_muufl"
    out.mkdir(exist_ok=True)
    results = {}
    for name in SCENES:
        cube, keep, gt = load(name)
        X = dw.whitened_pca(cube, keep, D)
        res = results[name] = {"cube global": dw.evaluate((X ** 2).sum(0), keep, gt)}
        scores, fb = dw.dual_window_scores(X, keep, WINDOWS)
        for (i, o), s in scores.items():
            res[f"cube dw {i}/{o}"] = dw.evaluate(s, keep, gt) | {"fallback": fb[(i, o)]}
        print(name, "done", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
