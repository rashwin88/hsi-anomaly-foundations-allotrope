"""
How separable are natural vs man-made, and the man-made classes from each other, with a strong classical matcher?
No training - this bounds what the physics and the library allow before we design classes.

For each sensor (PRISMA bench scenes; EnMAP / AVIRIS-NG implant-bench hosts) the 411-spectrum family library is
resampled to the scene's bands. Each library spectrum t is mixed into real background pixels at fill a:

    x = g * (a t + (1 - a) b) + n          g: smooth multiplicative atmospheric gain, |g - 1| <= gain_err
    r = g * ring mean around the pixel     (the same atmosphere), n: per-band noise from the scene

and matched by ACE on d = x - r against every OTHER library spectrum's direction s_j = t_j - r, whitened with the
scene's own covariance (top-k PCA + ridge). Leave-one-spectrum-out: a spectrum is never matched against itself.
Predicted family = the family of the best-matching spectrum; level 1 = its group (natural / man-made).

    python research/experiments/spectral-namer/separability.py [--per-entry 12] [--gain 0.1]
Writes separability_<sensor>.json and prints level-1 accuracy and the man-made confusion per fill.
"""

import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from app.spectral_match.library import load_slim_bundle, build_sensor_cache       # noqa: E402
from app.spectral_match.family_library import tag_library                         # noqa: E402
from app.evaluation.datasets import PrismaBench, load_implant_host                 # noqa: E402

FILLS = (0.1, 0.25, 0.5, 1.0)
OUT = Path(__file__).resolve().parent


def scenes(sensor: str):
    if sensor == "prisma":
        for sc in PrismaBench("C:/Users/FITT/Documents/stage1_datasets"):
            yield sc.name, sc.cube.astype(np.float32), np.asarray(sc.wavelengths_nm, float), sc.valid, sc.objects > 0
    else:
        for host in sorted((Path("C:/es/implant/bench-v2.1") / sensor).iterdir())[:3]:
            cube, wl, valid = load_implant_host(host)
            gt = np.load(host / "gt.npy") > 0 if (host / "gt.npy").exists() else np.zeros(valid.shape, bool)
            yield host.name, cube, np.asarray(wl, float), valid, gt


def library_for(wl: np.ndarray, cache_dir: Path):
    fwhm = np.abs(np.gradient(wl)) * 1.1
    bundle = load_slim_bundle(REPO / "data" / "family_library")
    entries, _ = build_sensor_cache(bundle=bundle, sensor_id=f"sep{len(wl)}", target_wl_nm=wl, target_fwhm_nm=fwhm,
                                    cache_dir=cache_dir, min_coverage=0.9, progress=False)
    lib = tag_library(entries)
    good = np.all(np.isfinite(lib.refl), axis=0)                     # bands every entry covers
    keep = np.isfinite(lib.refl[:, good]).all(1)
    return lib.subset(np.flatnonzero(keep)), good


def scene_stats(cube, valid, k=30, ridge=1e-3, n=60000, seed=0):
    rng = np.random.default_rng(seed)
    X = cube[:, valid].T
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)].astype(np.float64)
    C = np.cov((X - X.mean(0)).T)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    floor = max(w[k], 1e-12) + ridge * w[0]
    Winv = V @ np.diag(1.0 / np.maximum(w, floor)) @ V.T             # top-k as measured, the rest floored
    diff = (cube[:, 1:, :] - cube[:, :-1, :])[:, valid[1:, :] & valid[:-1, :]]
    sigma = diff.std(1) / np.sqrt(2)                                  # shift-difference noise per band
    return Winv, sigma


def ring_mean(cube, valid, i, j, inner=2, outer=7):
    i0, i1, j0, j1 = max(0, i - outer), i + outer + 1, max(0, j - outer), j + outer + 1
    win = cube[:, i0:i1, j0:j1]; v = valid[i0:i1, j0:j1].copy()
    ci, cj = i - i0, j - j0
    v[max(0, ci - inner):ci + inner + 1, max(0, cj - inner):cj + inner + 1] = False
    return win[:, v].mean(1) if v.any() else cube[:, i, j]


def run(sensor: str, per_entry: int, gain_err: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    results = {f: {"y": [], "p": []} for f in FILLS}
    for name, cube, wl, valid, objects in scenes(sensor):
        lib, good = library_for(wl, OUT / "cache")
        cube = cube[good]; wlg = wl[good]
        Winv, sigma = scene_stats(cube, valid)
        T = lib.refl[:, good].astype(np.float64)                     # (N, B)
        fams = np.array(lib.family); groups = np.array(lib.group)
        bg_ok = valid & ~ndimage.binary_dilation(objects, iterations=8)
        bg_ok[:8] = bg_ok[-8:] = False; bg_ok[:, :8] = bg_ok[:, -8:] = False
        ii, jj = np.nonzero(bg_ok)
        x_band = np.linspace(0, 1, len(wlg))
        for e in range(len(T)):
            for _ in range(per_entry):
                k = rng.integers(len(ii)); i, j = ii[k], jj[k]
                b = cube[:, i, j].astype(np.float64); r = ring_mean(cube, valid, i, j).astype(np.float64)
                g = 1 + gain_err * np.clip(np.polyval(rng.normal(0, 0.5, 3), x_band), -1, 1)   # smooth gain curve
                for a in FILLS:
                    x = g * (a * T[e] + (1 - a) * b) + rng.normal(0, 1, len(b)) * sigma
                    d = x - g * r
                    S = T - g * r                                     # (N, B) candidate directions
                    num = (S @ Winv @ d) ** 2
                    den = np.einsum("nb,bc,nc->n", S, Winv, S) * (d @ Winv @ d) + 1e-30
                    ace = num / den
                    ace[e] = -1                                       # leave this spectrum out
                    results[a]["y"].append(e); results[a]["p"].append(int(np.argmax(ace)))
        print(f"{sensor} {name}: {len(T)} spectra x {per_entry} backgrounds, {len(wlg)} bands", flush=True)
        break_after_one = sensor != "prisma"
        if break_after_one:
            pass
    out = {}
    for a in FILLS:
        y, p = np.array(results[a]["y"]), np.array(results[a]["p"])
        gy, gp, fy, fp = groups[y], groups[p], fams[y], fams[p]
        mm = gy == "manmade"
        lvl1_mm = float((gp[mm] == "manmade").mean()); lvl1_nat = float((gp[~mm] == "natural").mean())
        fam_list = sorted(set(fams[groups == "manmade"]))
        conf = {f: {g2: int(((fy == f) & (fp == g2)).sum()) for g2 in sorted(set(fams))} for f in fam_list}
        lvl2 = float((fp[mm] == fy[mm]).mean())
        lvl2_given_mm = float((fp[mm & (gp == "manmade")] == fy[mm & (gp == "manmade")]).mean())
        out[str(a)] = dict(level1_manmade_called_manmade=lvl1_mm, level1_natural_called_natural=lvl1_nat,
                           level2_correct_family=lvl2, level2_correct_given_level1_right=lvl2_given_mm,
                           confusion_manmade_rows=conf, n=int(len(y)))
        print(f"  fill {a:>4}: L1 man-made->man-made {lvl1_mm:.2f}, natural->natural {lvl1_nat:.2f} | "
              f"L2 family {lvl2:.2f} (given L1 right {lvl2_given_mm:.2f})", flush=True)
    (OUT / f"separability_{sensor}.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-entry", type=int, default=12)
    ap.add_argument("--gain", type=float, default=0.1)
    ap.add_argument("--sensors", nargs="+", default=["prisma", "enmap", "aviris_ng"])
    args = ap.parse_args()
    for s in args.sensors:
        run(s, args.per_entry, args.gain)
