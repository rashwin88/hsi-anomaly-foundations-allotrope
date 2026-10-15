"""
Step 4: does keeping the ring inside one land-cover type rescue dual-window on PRISMA?

Steps 1-3 left one explanation standing for why dual-window loses on PRISMA
but helps on the small airborne benchmarks: a full PRISMA scene is full of
land-cover edges, and a ring that straddles one makes the edge look
anomalous. This step scores the Indradhanu residual three new ways, each at
WINDOWS:

  - seg-product: the ring only counts pixels of the centre pixel's class from
    the product's scene_segmentation rules (water / cloud / shadow /
    vegetation / other; app/utils/spectral_indices/scene_indices.py).
  - seg-kmeans: the same with K_CLUSTERS k-means land-cover clusters fitted
    on the cube's 30 whitened PCs: finer classes, in the spirit of
    cluster-based anomaly detection.
  - local-mean: ring mean, global covariance (no local covariance at all).

It also runs the product's classical detector, MNF-RX
(app/detectors/mnf_compression_detector.py, 10 components), on PRISMA and on
HAD100, so "Indradhanu + Mahalanobis" is compared with what ships today.

Writes OUT_ROOT/step4.json (PRISMA) and dual_window_public/had100_mnf.json.

Gotchas:

  - MNF-RX gets the same filled 165-band cube Indradhanu sees, as it would
    in the product after band_filter_apply.
  - On HAD100, MNF-RX gets the radiance-mapped 165-band cube from step 3.
"""
from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "indradhanu-fp-filter"))
import dw_core as dw  # noqa: E402
import step1_prisma as p1  # noqa: E402
import step1_candidates as s1  # noqa: E402

WINDOWS = [(7, 21), (11, 31), (15, 41), (21, 61)]
K_CLUSTERS = 8


def mnf_rx(cube: np.ndarray, validity: np.ndarray, wavelengths) -> np.ndarray:
    """The product's MNF-RX score map (NaN outside its spatial mask → 0)."""
    from app.detectors.mnf_compression_detector import MNFCompressionDetector
    vend = types.SimpleNamespace(normalized_hyperspectral_cube=cube, validity_cube=validity,
                                 band_cw_order=list(wavelengths),
                                 band_validity_by_position=[1] * len(wavelengths))
    det = MNFCompressionDetector(vend)
    det.fit()
    return np.nan_to_num(det.detect(cube, validity)).astype(np.float32)


def product_classes(cube: np.ndarray, validity: np.ndarray, wavelengths) -> np.ndarray:
    """0 other, 1 water, 2 cloud, 3 shadow, 4 vegetation, first match wins in that order."""
    from app.utils.spectral_indices.scene_indices import compute_scene_segmentation
    seg = compute_scene_segmentation(cube, validity, np.asarray(wavelengths, dtype=np.float64))
    lab = np.zeros(cube.shape[1:], np.int8)
    for code, m in ((4, seg.mask_vegetation), (3, seg.mask_shadow), (2, seg.mask_cloud), (1, seg.mask_water)):
        lab[m.astype(bool)] = code
    return lab


def kmeans_classes(Xc: np.ndarray, keep: np.ndarray) -> np.ndarray:
    P = Xc[:, keep].T
    km = MiniBatchKMeans(K_CLUSTERS, random_state=0, n_init=3, batch_size=4096)
    km.fit(P[np.random.default_rng(0).choice(len(P), min(len(P), 200_000), replace=False)])
    lab = np.full(keep.shape, -1, np.int16)
    lab[keep] = km.predict(P)
    return lab


def run_prisma() -> None:
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as grid
    results = {}
    for n, (he5, _) in enumerate(s1.find_scenes(), start=1):
        d1 = s1.OUT_ROOT / f"set{n}"
        keep, gt = np.load(d1 / "keep.npy"), np.load(d1 / "gt.npy")
        cube, _ = s1.build_vendable_cube(he5)
        validity = np.broadcast_to(keep[None], cube.shape).astype(np.int8)
        res = results[f"set{n}"] = {"MNF-RX (product)": dw.evaluate(mnf_rx(cube, validity, grid), keep, gt)}
        labels = {"seg-product": product_classes(cube, validity, grid)}
        del cube
        labels["seg-kmeans"] = kmeans_classes(np.load(p1.OUT_ROOT / f"set{n}_cube_pca{p1.D}.npy"), keep)
        X = np.load(p1.OUT_ROOT / f"set{n}_residual_pca{p1.D}.npy")
        res["residual global"] = dw.evaluate((X ** 2).sum(0), keep, gt)
        plain, _ = dw.dual_window_scores(X, keep, WINDOWS)
        variants = {"dw": plain, "local-mean": dw.local_mean_scores(X, keep, WINDOWS)}
        for name, lab in labels.items():
            variants[name] = dw.segmented_dual_window_scores(X, keep, lab, WINDOWS)
            res[f"{name} class share"] = {int(c): round(float((lab[keep] == c).mean()), 3)
                                          for c in np.unique(lab[keep])}
        for name, maps in variants.items():
            for (i, o), s in maps.items():
                res[f"residual {name} {i}/{o}"] = dw.evaluate(s, keep, gt)
        print(f"set{n} done", flush=True)
        (p1.OUT_ROOT / "step4.json").write_text(json.dumps(results, indent=1))


def run_had100_mnf() -> None:
    import step3_public as p3
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as grid
    stats = json.loads((s1.REPO / "app" / "constants" / "hyperspectral.json").read_text())
    t_mu, t_sd = np.array(stats["mean"])[:, None, None], np.array(stats["std"])[:, None, None]
    g = np.asarray(grid, dtype=np.float64)
    tiles = list(p3.had100_scenes())
    gridded = []
    for _, c, wl, _ in tiles:   # same resampling and pooled mapping as step 3
        i1 = np.clip(np.searchsorted(wl, g), 1, len(wl) - 1)
        w = ((g - wl[i1 - 1]) / (wl[i1] - wl[i1 - 1]))[:, None, None]
        c = np.nan_to_num(c)
        gridded.append(((1 - w) * c[i1 - 1] + w * c[i1]).astype(np.float32))
    pooled = np.concatenate([x.reshape(len(g), -1) for x in gridded], 1)
    l_mu, l_sd = pooled.mean(1)[:, None, None], pooled.std(1)[:, None, None] + 1e-9
    results = {}
    for (name, cube, _, gt), x in zip(tiles, gridded):
        keep = np.abs(cube).sum(0) > 0
        refl = (t_mu + t_sd * (x - l_mu) / l_sd).astype(np.float32)
        validity = np.broadcast_to(keep[None], refl.shape).astype(np.int8)
        results[name] = {"MNF-RX (product)": dw.evaluate(mnf_rx(refl, validity, grid), keep, gt, curve=p3.CURVE)}
    (p3.OUT / "had100_mnf.json").write_text(json.dumps(results, indent=1))
    print("had100 mnf done", flush=True)


if __name__ == "__main__":
    logging.disable(logging.INFO)
    stages = sys.argv[1:] or ["had100", "prisma"]
    if "had100" in stages:
        run_had100_mnf()
    if "prisma" in stages:
        run_prisma()
