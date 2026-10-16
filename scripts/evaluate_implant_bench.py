"""
Score Indradhanu-family checkpoints on the implanted-target benchmark (scripts/build_implant_bench.py,
docs/lld/implant-bench.md).

    python scripts/evaluate_implant_bench.py <bench dir> NAME=checkpoint [NAME=checkpoint ...] [--out results.json]

Per host scene: the product's band exclusions, then each model's own reading - wavelength stems read the host's
bands that their trained-wavelength record covers (as scripts/evaluate_prisma_bench.py's native mode); the
fixed-grid hyperspectral_segformer_mae (v0.2.0) reads a linear resample onto the 165-band common grid (close to,
not identical with, the product's band_filter_apply). Missing voxels are filled per band from the nearest valid
pixel. Scores: residual Mahalanobis (default) and combined.

Reported per model, overall and by tier (realised contrast), source (signature / material) and sensor: objects caught
in the top 1% and 0.1% of scores (1-pixel tolerance), and pAUC <= 1% FPR with that group's pixels as positives and
non-object pixels as negatives. Reconstructions are cached under <bench>/recon/ (float16).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from app.utils.evaluation.implant import TIERS  # noqa: E402
from app.utils.evaluation.prisma_bench import detection_measures  # noqa: E402

EXCLUSIONS = [(0, 450), (912, 978), (1131, 1152), (1350, 1450), (1800, 1950)]   # the product's default
NODATA, SCALE = -32768, 1e-4


def load_host(host: Path):
    """(refl (B, H, W) float32 with gaps filled, wl (B,), spatial_valid (H, W) bool, gt (H, W) int32, objects)."""
    cube = np.load(host / "cube.npy")
    wl = np.load(host / "wavelengths.npy")
    keep_band = np.array([not any(lo <= w <= hi for lo, hi in EXCLUSIONS) for w in wl])
    cube, wl = cube[keep_band], wl[keep_band]
    finite = cube != NODATA
    spatial = finite.mean(0) >= 0.9
    good = finite[:, spatial].mean(1) >= 0.6                     # drop bands mostly empty over valid pixels
    cube, wl, finite = cube[good], wl[good], finite[good]
    refl = cube.astype(np.float32) * SCALE
    for b in range(len(wl)):                                      # per-band nearest-valid fill of missing voxels
        hole = ~finite[b] & spatial
        if hole.any():
            idx = ndimage.distance_transform_edt(~finite[b], return_distances=False, return_indices=True)
            refl[b][hole] = refl[b][idx[0][hole], idx[1][hole]]
    refl[:, ~spatial] = 0.0
    return refl, wl, spatial, np.load(host / "gt.npy"), json.load(open(host / "objects.json"))


def reconstruct(label: str, ckpt: str, refl: np.ndarray, wl: np.ndarray, spatial: np.ndarray, cache: Path):
    """(cube the model read, its reconstruction), cached."""
    import torch
    from evaluate_prisma_bench import build
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
    inf = build(label, ckpt)
    if isinstance(inf, SpectralStemInferencer):
        bm = inf.model.inner.trained_band_mask(wl).cpu().numpy()
        cube = refl[bm]
        inf.set_wavelengths(wl[bm])
    else:                                                         # fixed 165-band grid: linear resample
        from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as grid
        grid = np.asarray(grid, float)
        M = np.stack([np.interp(grid, wl, np.eye(len(wl))[i]) for i in range(len(wl))], 1)   # (165, B)
        cube = np.einsum("gb,bhw->ghw", M, refl, optimize=True).astype(np.float32)
    if cache.exists():
        return cube, np.load(cache).astype(np.float32)
    with torch.no_grad():
        recon = inf.predict_full_scene(torch.from_numpy(cube), torch.from_numpy(spatial[None]).float())
    recon = recon.detach().cpu().numpy().astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, recon.astype(np.float16))
    return cube, recon


def object_hits(score: np.ndarray, keep: np.ndarray, gt: np.ndarray, top_pct: float) -> set:
    k = keep & np.isfinite(score)
    hits = ndimage.binary_dilation((score >= np.percentile(score[k], 100 - top_pct)) & k, iterations=1)
    return set(np.unique(gt[hits & (gt > 0)]).tolist())


def score_host(score, keep, gt, objects) -> list:
    """Per object: id, tier, source, hit flags at 1% and 0.1%."""
    h1, h01 = object_hits(score, keep, gt, 1.0), object_hits(score, keep, gt, 0.1)
    return [dict(id=o["id"], tier=o["tier"], source=o["source"], hit1=o["id"] in h1, hit01=o["id"] in h01) for o in objects]


def group_pauc(score, keep, gt, ids) -> float | None:
    k = keep & np.isfinite(score)
    pos = np.isin(gt, list(ids)) & k
    neg = (gt == 0) & k
    sel = pos | neg
    m = detection_measures(score[sel], pos[sel])
    return m.get("pauc_1pct")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    p.add_argument("bench")
    p.add_argument("models", nargs="+", help="NAME=checkpoint")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    from app.utils.anomaly_detection.scoring import compute_score
    bench = Path(a.bench)
    manifest = json.load(open(bench / "manifest.json"))
    out_path = Path(a.out or bench / "results.json")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for spec in a.models:
        label, ckpt = spec.split("=", 1)
        for h in manifest["hosts"]:
            key = f"{h['sensor']}/{h['scene']}"
            if key in results.get(label, {}):
                continue
            host = bench / h["sensor"] / h["scene"]
            refl, wl, spatial, gt, objects = load_host(host)
            cube, recon = reconstruct(label, ckpt, refl, wl, spatial, bench / "recon" / label / f"{h['scene']}.npy")
            entry = {}
            for method in ("residual_mahalanobis", "combined"):
                s = compute_score(cube, recon, spatial.astype(np.uint8), method=method)
                entry[method] = {"objects": score_host(s, spatial, gt, objects),
                                 "pauc": {t: group_pauc(s, spatial, gt, [o["id"] for o in objects if o["tier"] == t])
                                          for t in TIERS}}
            results.setdefault(label, {})[key] = entry
            out_path.write_text(json.dumps(results))
            print(label, key, flush=True)
    summarise(results, manifest)


def summarise(results: dict, manifest: dict) -> None:
    for method in ("residual_mahalanobis", "combined"):
        print(f"\n== {method}: objects caught top 1% / top 0.1% (share), mean pAUC <= 1% FPR over hosts")
        for label, per in results.items():
            objs = [(k.split("/")[0], o) for k, e in per.items() for o in e[method]["objects"]]
            def share(sel, key):
                xs = [o[key] for s, o in objs if sel(s, o)]
                return f"{sum(xs) / max(len(xs), 1):.0%}" if xs else "-"
            cells = [f"all {share(lambda s, o: True, 'hit1')}/{share(lambda s, o: True, 'hit01')}"]
            for t in TIERS:
                pa = [e[method]["pauc"][t] for e in per.values() if e[method]["pauc"][t] is not None]
                cells.append(f"{t} {share(lambda s, o: o['tier'] == t, 'hit1')}/{share(lambda s, o: o['tier'] == t, 'hit01')}"
                             f" pAUC {np.mean(pa):.3f}" if pa else f"{t} -")
            for src in ("signature", "material"):
                cells.append(f"{src} {share(lambda s, o: o['source'] == src, 'hit1')}")
            for sensor in ("enmap", "prisma", "aviris_ng"):
                cells.append(f"{sensor} {share(lambda s, o: s == sensor, 'hit1')}")
            print(f"{label:12s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
