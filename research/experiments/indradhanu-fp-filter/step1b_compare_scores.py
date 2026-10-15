"""
Step 1b: compare three ways of scoring the four PRISMA scenes at equal cuts.

  - Indradhanu L1 — the product's reconstruction-error score (from step 1).
  - Indradhanu residual-Mahalanobis — RX run on the residual (cube − recon)
    in its top-30 principal components, the S2DWMTrans post-processing idea.
  - PCA30-RX — plain global RX on the cube's top-30 principal components,
    as a classical baseline and a check that the GT is aligned.

Run after step1_candidates.py (it reads its gt.npy / score_L1.npy). Writes
score_residual_maha.npy and score_pca30_rx.npy next to them for step 2.

Gotchas:

  - PCA and covariance are fitted on strided subsamples (every 20th / 5th
    valid pixel) to keep memory and time down; the scores themselves are
    computed for every valid pixel.
  - Plain RX on all 165 bands is deliberately avoided (near-singular
    covariance, see resolver.py), hence the 30-component projection.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step1_candidates as s1  # noqa: E402

CUTS = (99.5, 99.0, 98.0, 95.0)


def pca_mahalanobis(X: np.ndarray, k: int = 30) -> np.ndarray:
    """Squared Mahalanobis distance of each row of X in its top-k principal components."""
    X = X - X.mean(0)
    _, _, vt = np.linalg.svd(X[::20], full_matrices=False)
    P = X @ vt[:k].T
    cov_inv = np.linalg.inv(np.cov(P[::5].T))
    return np.einsum("ij,jk,ik->i", P, cov_inv, P)


def main() -> None:
    logging.disable(logging.INFO)
    inferencer = s1.load_indradhanu()
    results: dict[str, dict] = {}
    for n, (he5, _) in enumerate(s1.find_scenes(), start=1):
        out = s1.OUT_ROOT / f"set{n}"
        cube, keep = s1.build_vendable_cube(he5)
        gt = np.load(out / "gt.npy")
        with torch.no_grad():
            recon = inferencer.predict_full_scene(
                torch.from_numpy(cube), torch.from_numpy(keep[None].astype(np.float32))
            ).numpy()

        maps = {"Indradhanu L1": np.load(out / "score_L1.npy")}
        for name, fname, data in (
            ("Indradhanu residual-Mahalanobis", "score_residual_maha.npy", cube - recon),
            ("PCA30-RX", "score_pca30_rx.npy", cube),
        ):
            score = np.zeros(keep.shape, dtype=np.float32)
            score[keep] = pca_mahalanobis(data[:, keep].T.astype(np.float64))
            np.save(out / fname, score)
            maps[name] = score

        for name, score in maps.items():
            r = results.setdefault(name, {"auc": [], "cuts": {p: [0, 0, 0] for p in CUTS}})
            r["auc"].append(roc_auc_score(gt[keep], score[keep]))
            for p in CUTS:
                c = s1.recall_at_cut(score, keep, gt, p)
                t = r["cuts"][p]
                t[0] += c["objects_caught"]
                t[1] += c["objects_total"]
                t[2] += c["false_positive_px"]
        del cube, recon

    for name, r in results.items():
        print(name, "AUC per scene:", " ".join(f"{a:.3f}" for a in r["auc"]))
        for p, (hit, tot, fp) in r["cuts"].items():
            print(f"   top {100 - p:.1f}%: objects {hit}/{tot} ({100 * hit / tot:.0f}%), FP px {fp}")


if __name__ == "__main__":
    main()
