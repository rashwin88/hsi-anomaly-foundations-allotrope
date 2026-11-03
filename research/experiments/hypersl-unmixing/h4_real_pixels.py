"""
H4 - N002 and the training-free arms on REAL public pixels (Jasper, Urban4), labels from the ground-truth abundances
mapped to the namer's taxonomy (README table). Inference only.

Per pixel: dominant endmember (pure: max abundance >= 0.8, mixed: 0.5-0.8). Level 1: man-made vs natural score
(namer: level-1 logit; others: best man-made class score - natural class score), AUC and recall at 5% natural false
alarms. Level 2: share of man-made-dominant pixels of a mappable class (road / asphalt -> concrete_asphalt) whose best
man-made class is right. The whole library is used (no spectrum is "own" here).
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as c
import h5_neighbourhood as h5

MAP = {"Jasper": {"tree": "natural", "water": "natural", "soil": "natural", "road": "concrete_asphalt"},
       "Urban4": {"asphalt": "concrete_asphalt", "grass": "natural", "tree": "natural", "roof": "manmade_l1"}}
CLASSES = ("oil_fuel", "plastic", "textile_fabric", "wood_paper", "paint_coatings", "metal", "concrete_asphalt")


def main():
    from app.spectral_match.namer.bank import build_bank
    from app.spectral_match.namer.infer import load_namer
    namer = load_namer(h5.N002, c.DEV)
    rng = np.random.default_rng(1)
    res = {}
    for name, cube, wl in h5.scenes():
        if name not in MAP:
            continue
        s = c.load(name)
        A = s.A.reshape(len(s.E), s.H, s.W)
        ring, _, _ = h5.ring_and_noise(cube)
        H, W, B = cube.shape
        yy, xx = np.mgrid[7:H - 7, 7:W - 7]
        rr, cc = yy.ravel(), xx.ravel()
        if len(rr) > 6000:
            k = rng.choice(len(rr), 6000, replace=False); rr, cc = rr[k], cc[k]
        dom = A[:, rr, cc].argmax(0); amax = A[:, rr, cc].max(0)
        keep = amax >= 0.5
        rr, cc, dom, amax = rr[keep], cc[keep], dom[keep], amax[keep]
        lab = np.array([MAP[name][s.em_names[k]] for k in dom])
        is_mm = lab != "natural"
        X, R = cube[rr, cc], ring[rr, cc]
        bank = build_bank(wl, h5.CACHE, h5.LIB_ROOT)
        Lr = bank.refl.numpy().astype(np.float64)
        lib_cls = np.where(bank.cls.numpy() < 0, h5.C, bank.cls.numpy())
        f = c.Frame(cube.reshape(-1, B))
        zL = c.unit(c.encode(f(Lr), wl))
        flat = cube.reshape(-1, B)
        mu = flat.mean(0)
        cls_scores = {}
        def per_class(sc):
            return np.stack([sc[:, lib_cls == k].max(1) for k in range(h5.C + 1)], 1)
        cls_scores["sam"] = per_class(c.unit(X) @ c.unit(Lr).T)
        for lvl, bh in (("ls_none", np.zeros_like(X)), ("ls_mean", np.repeat(mu[None], len(X), 0)), ("ls_ring", R)):
            cls_scores[lvl] = per_class(-h5.residual_scores(X, bh, Lr)[1])
        cls_scores["lat_none"] = per_class(c.unit(c.encode(f(X), wl)) @ zL.T)
        out = {}
        for tag, rm in (("namer_ring", R), ("namer_self", X)):
            dev = c.DEV
            w = torch.as_tensor(wl, dtype=torch.float32, device=dev)
            with torch.no_grad():
                lib = namer.forward_library(bank.refl.to(dev), w)
                lc = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, h5.C)).to(dev)
                z, _, _ = namer.forward_pixel(torch.as_tensor(X, dtype=torch.float32, device=dev), torch.as_tensor(rm, dtype=torch.float32, device=dev), w)
                sc = namer.class_scores(z, lib, lc, h5.C + 1)
                l1 = namer.level1_logit(sc).cpu().numpy()
            out[tag] = (l1, sc[:, :h5.C].argmax(1).cpu().numpy())
        for m, cs in cls_scores.items():
            out[m] = (cs[:, :h5.C].max(1) - cs[:, h5.C], cs[:, :h5.C].argmax(1))
        r = {"n": int(len(X)), "n_manmade": int(is_mm.sum()), "n_natural": int((~is_mm).sum()),
             "n_by_label": {k: int((lab == k).sum()) for k in np.unique(lab)}, "methods": {}}
        pure = amax >= 0.8
        for m, (l1, c2) in out.items():
            rec = {}
            for stratum, sel in (("all", np.ones_like(pure)), ("pure", pure), ("mixed", ~pure)):
                y = is_mm[sel]
                if y.all() or (~y).all():
                    continue
                cut = np.quantile(l1[sel][~y], 0.95)
                l2 = sel & (lab == "concrete_asphalt")
                rec[stratum] = {"auc": round(float(roc_auc_score(y, l1[sel])), 3),
                                "recall_at_5pct_fa": round(float((l1[sel][y] > cut).mean()), 3),
                                "l2_concrete_acc": round(float((c2[l2] == CLASSES.index("concrete_asphalt")).mean()), 3) if l2.any() else None,
                                "n_l2": int(l2.sum())}
            if m.startswith("namer"):
                rec["called_manmade_at_0"] = {"manmade": round(float((l1[is_mm] > 0).mean()), 3), "natural": round(float((l1[~is_mm] > 0).mean()), 3)}
            r["methods"][m] = rec
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open("results/h4_real.json", "w"), indent=1)


if __name__ == "__main__":
    main()
