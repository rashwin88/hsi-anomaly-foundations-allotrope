"""
Step 6: HAD100 under its official protocol, to place Indradhanu + Mahalanobis
against the published leaderboard (Li et al., "You Only Train Once", IEEE TGRS
2023, Table V; arXiv 2303.18001).

The protocol, rebuilt from HAD100/main.py:

  - 100 test images: each AVIRIS-NG target tile cropped to 64×64, at the
    offsets in main.py's crop_dict (six tiles give two crops); other tiles
    use their top-left 64×64.
  - The 276 "good" bands of main.py's band_select, then the FIRST 50 / 100 /
    200 of those.
  - mAUC = the mean over the 100 images of the per-image pixel ROC-AUC.

Classical scorers run under each band setting. GRX is the calibration check:
it should land near the paper's 0.9799 / 0.9714 / 0.9649. Indradhanu cannot
take a band subset, because it needs its 165-band 460-2450 nm grid. It runs
once on the full spectrum, as in step 3, and is reported against all three
settings with that caveat.

Gotchas:

  - GRX uses the pseudo-inverse of the full-band covariance, as plain RX does.
  - main.py's crop_dict is copied verbatim below.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "indradhanu-fp-filter"))
import dw_core as dw  # noqa: E402
import step3_public as p3  # noqa: E402

GOOD_BANDS = np.r_[15:109, 118:145, 158:187, 227:274, 328:407]
BAND_SETTINGS = (50, 100, 200)
CROP = {'ang20170821t183707_102': [(0, 15)],
        'ang20170821t183707_124': [(0, 0)],
        'ang20170821t195229_21': [(99 - 63, 99 - 63)], 'ang20170825t173426_14': [(0, 0)],
        'ang20170825t173426_17': [(0, 99 - 63)],
        'ang20170908t225309_32': [(0, 0), (99 - 63, 80 - 63)], 'ang20170908t225309_34': [(99 - 63, 0)],
        'ang20170908t225309_36': [(0, 0), (10, 99 - 63)], 'ang20170908t225309_37': [(0, 0), (10, 99 - 63)],
        'ang20170908t225309_39': [(0, 79 - 63)], 'ang20171012t194435_23': [(0, 0)],
        'ang20180227t082814_15': [(0, 0), (99 - 63, 99 - 63)],
        'ang20180227t082814_17': [(0, 99 - 63), (99 - 63, 5)],
        'ang20191004t185054_22': [(0, 0)], 'ang20191004t185054_27': [(0, 3)],
        'ang20210614t141018_32': [(0, 80 - 63)],
        'ang20210614t141018_37': [(10, 0), (99 - 63, 99 - 63)], 'ang20210614t141018_38': [(99 - 63, 0)]}
LOCAL = [(5, 15)]
DW = [(7, 11)]


def grx(cube: np.ndarray) -> np.ndarray:
    X = cube.reshape(cube.shape[0], -1).T.astype(np.float64)
    X = X - X.mean(0)
    return np.einsum("ij,jk,ik->i", X, np.linalg.pinv(np.cov(X.T)), X).reshape(cube.shape[1:])


def crops():
    """(name, full-tile radiance cube, wavelengths, (x, y)) for the 100 test images."""
    for name, cube, wl, gt in p3.had100_scenes():
        offsets = CROP.get(name, [(0, 0)])
        for i, (x, y) in enumerate(offsets):
            yield (f"{name}_{i + 1}" if len(offsets) > 1 else name), cube, wl, gt, x, y


def main() -> None:
    import step1_candidates as s1
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
    from app.utils.anomaly_detection.scoring import compute_score
    g = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
    stats = json.loads((s1.REPO / "app" / "constants" / "hyperspectral.json").read_text())
    t_mu, t_sd = np.array(stats["mean"])[:, None, None], np.array(stats["std"])[:, None, None]

    def to_grid(c, wl):
        i1 = np.clip(np.searchsorted(wl, g), 1, len(wl) - 1)
        w = ((g - wl[i1 - 1]) / (wl[i1] - wl[i1 - 1]))[:, None, None]
        return ((1 - w) * c[i1 - 1] + w * c[i1]).astype(np.float32)

    # radiance → model statistics: the same pooled mapping as step 3 (full tiles)
    pooled = np.concatenate([to_grid(np.nan_to_num(c), wl).reshape(len(g), -1)
                             for _, c, wl, _ in p3.had100_scenes()], 1)
    l_mu, l_sd = pooled.mean(1)[:, None, None], pooled.std(1)[:, None, None] + 1e-9
    del pooled
    inferencer = s1.load_indradhanu()

    results = {}
    for name, cube, wl, gt_full, x, y in crops():
        cube = np.nan_to_num(cube)
        c = cube[:, x:x + 64, y:y + 64]
        gt = gt_full[x:x + 64, y:y + 64]
        keep = np.ones(gt.shape, bool)
        res = results[name] = {"gt_px": int(gt.sum())}
        auc = lambda s: float(roc_auc_score(gt.ravel(), s.ravel()))  # noqa: E731
        for nb in BAND_SETTINGS:
            sub = c[GOOD_BANDS[:nb]]
            res[f"GRX {nb}"] = auc(grx(sub))
            X = dw.whitened_pca(sub, keep, min(30, nb))
            res[f"PCA30-RX {nb}"] = auc((X ** 2).sum(0))
            res[f"local-mean 5/15 {nb}"] = auc(dw.local_mean_scores(X, keep, LOCAL)[LOCAL[0]])
            res[f"dual-window 7/11 {nb}"] = auc(dw.dual_window_scores(X, keep, DW)[0][DW[0]])

        refl = (t_mu + t_sd * (to_grid(cube, wl) - l_mu) / l_sd).astype(np.float32)[:, x:x + 64, y:y + 64]
        pad = np.pad(refl, ((0, 0), (32, 32), (32, 32)), mode="reflect")
        with torch.no_grad():
            recon = inferencer.predict_full_scene(torch.from_numpy(pad), torch.ones(1, 128, 128)).cpu().numpy()
        recon = recon[:, 32:96, 32:96]
        res["Indradhanu L1 (full spectrum)"] = auc(compute_score(refl, recon, keep, method="L1"))
        R = dw.whitened_pca(refl - recon, keep, 30)
        res["Indradhanu + Mahalanobis (full spectrum)"] = auc((R ** 2).sum(0))
        res["Indradhanu + local-mean 5/15 (full spectrum)"] = auc(dw.local_mean_scores(R, keep, LOCAL)[LOCAL[0]])
        X = dw.whitened_pca(c[GOOD_BANDS], keep, 30)
        res["PCA30-RX (all 276 bands)"] = auc((X ** 2).sum(0))
        res["GRX (all 276 bands)"] = auc(grx(c[GOOD_BANDS]))
    print(len(results), "test images")
    (p3.OUT / "had100_protocol.json").write_text(json.dumps(results, indent=1))
    keys = [k for k in next(iter(results.values())) if k != "gt_px"]
    for k in keys:
        print(f"{k:46s} mAUC {np.mean([r[k] for r in results.values()]):.4f}")


if __name__ == "__main__":
    logging.disable(logging.INFO)
    main()
