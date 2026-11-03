"""
Step 1: HAD100's classical methods (GRX, LRX, CR, KCSR) on the 4 labelled PRISMA scenes, on exactly the cube the
evaluator's PCA-RX sees (app.evaluation.datasets.PrismaBench: native bands after the product exclusions, nearest-valid
filled reflectance, the evaluator's valid mask). Writes OUT/<scene>/<method>.npz (score float32) and OUT/timing.json.
Existing maps are skipped, so the run resumes.

    C:/av/Scripts/python.exe -X utf8 step1_classical.py [--methods grx,lrx,crd,kcsr] [--scenes set1,...]
    [--crop 256]   (a centre crop for a quick timing test; writes to OUT/_crop)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import methods as M  # noqa: E402
from app.evaluation.datasets import PrismaBench  # noqa: E402

ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT = ROOT / "had100_baselines"
CR_LAMS = (1e-3, 1e-2, 1e-1, 1.0)


def first_bands(wl: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n shortest-wavelength bands (HAD100's 'first n bands' of an ascending AVIRIS-NG axis)."""
    return np.sort(np.argsort(wl)[:n])


def run(method: str, scene, X, valid, d: Path, log: dict) -> None:
    def save(name, s, extra=None):
        np.savez_compressed(d / f"{name}.npz", score=s.astype(np.float32))
        log[name] = {"s": round(time.time() - t0, 1), **(extra or {})}
        print(scene.name, name, log[name], flush=True)

    t0 = time.time()
    if method == "grx":
        if not (d / "GRX.npz").exists():
            save("GRX", M.grx(X, valid))
        for n in (50, 100):
            if not (d / f"GRX_{n}b.npz").exists():
                t0 = time.time()
                save(f"GRX_{n}b", M.grx(X, valid, bands=first_bands(scene.wavelengths_nm, n)))
    elif method == "lrx":
        for inner, outer in ((13, 29), (5, 15)):
            name = f"LRX_{inner}_{outer}"
            if not (d / f"{name}.npz").exists():
                t0 = time.time()
                sc = M.lrx(X, valid, inner, outer)
                save(name, sc, {"float64_refits": M.lrx.last_refit})
    elif method in ("crd", "kcsr"):
        names = [f"CR_lam{lam:g}" for lam in CR_LAMS] if method == "crd" else ["KCSR_3_29"]
        if all((d / f"{n}.npz").exists() for n in names):
            return
        Z, kept = M.second_moment_projection(X, valid, 30)
        t0 = time.time()
        if method == "crd":
            res = M.crd(Z, valid, 5, 29, CR_LAMS)
            el = round(time.time() - t0, 1)
            for lam, s in res.items():
                np.savez_compressed(d / f"CR_lam{lam:g}.npz", score=s)
                log[f"CR_lam{lam:g}"] = {"s": el, "note": "all lambdas in one pass", "energy_kept_k30": kept}
            print(scene.name, "CR", el, kept, flush=True)
        else:
            save("KCSR_3_29", M.kcsr(Z, valid, 3, 29), {"energy_kept_k30": kept})
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--methods", default="grx,lrx,crd,kcsr")
    p.add_argument("--scenes", default="set1,set2,set3,set4")
    p.add_argument("--crop", type=int, default=0)
    a = p.parse_args()
    out = OUT / "_crop" if a.crop else OUT
    tpath = out / "timing.json"
    out.mkdir(parents=True, exist_ok=True)
    timing = json.loads(tpath.read_text()) if tpath.exists() else {}
    for scene in PrismaBench(str(ROOT), a.scenes.split(",")):
        X, valid = scene.cube, scene.valid
        if a.crop:
            H, W = valid.shape
            sl = (slice(H // 2 - a.crop // 2, H // 2 + a.crop // 2), slice(W // 2 - a.crop // 2, W // 2 + a.crop // 2))
            X, valid = np.ascontiguousarray(X[(slice(None),) + sl]), valid[sl].copy()
        d = out / scene.name
        d.mkdir(exist_ok=True)
        log = timing.setdefault(scene.name, {"shape": list(valid.shape), "bands": int(len(X)), "valid": int(valid.sum())})
        for m in a.methods.split(","):
            run(m, scene, X, valid, d, log)
            tpath.write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
