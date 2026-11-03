"""
N002 on the seven unmixing datasets (README "N002 on the unmixing datasets"). Inference only; N002 is a matcher, so it
is scored on naming, not abundances.

1. Endmember naming: GT and VCA endmember spectra through N002 (ring := the spectrum itself - an isolated spectrum
   has no neighbourhood) -> level-1 call, P(man-made), best class, P(class).
2. Per-pixel dominant-material naming against the GT abundances: pure (max abundance >= 0.9) and mixed (0.4-0.9).
   Level 1 (AUC, recall at 5% natural false alarms) where both man-made and natural pixels exist, else the share of
   natural pixels called man-made; level 2 (road / asphalt -> concrete_asphalt) accuracy at 100% / 50% coverage.
   N002 with the 3-7 px ring and with ring := pixel; comparators on the same pixels and library: SAM, least squares
   with b^ = 0 / scene mean / ring, ACE, reflectance mixture bank, HyperSL whole-pixel.
Scales: Jasper / 1e4 and Urban4 / 1e3 (reflectance); DC1-3, EGU as stored (reflectance); Samson as stored (normalised,
units unknown - README caveat).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as c
import h5_neighbourhood as h5

CLASSES = ("oil_fuel", "plastic", "textile_fabric", "wood_paper", "paint_coatings", "metal", "concrete_asphalt")
LABEL = {"Jasper": {"tree": "natural", "water": "natural", "soil": "natural", "road": "concrete_asphalt"},
         "Urban4": {"asphalt": "concrete_asphalt", "grass": "natural", "tree": "natural", "roof": "manmade_l1"},
         "Samson": {"soil": "natural", "tree": "natural", "water": "natural"}}
DEFAULT = "natural"                       # DC1-3 (USGS minerals), EGU (assumed natural): level 1 only


def label_of(name, em):
    return LABEL.get(name, {}).get(em, DEFAULT)


@torch.no_grad()
def namer_call(namer, X, R, wl, bank):
    dev = c.DEV
    w = torch.as_tensor(wl, dtype=torch.float32, device=dev)
    lib = namer.forward_library(bank.refl.to(dev), w)
    lc = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, h5.C)).to(dev)
    l1s, ps = [], []
    for i in range(0, len(X), 4096):
        z, _, _ = namer.forward_pixel(torch.as_tensor(X[i:i + 4096], dtype=torch.float32, device=dev),
                                      torch.as_tensor(R[i:i + 4096], dtype=torch.float32, device=dev), w)
        sc = namer.class_scores(z, lib, lc, h5.C + 1)
        l1s.append(namer.level1_logit(sc).cpu().numpy()); ps.append(sc[:, :h5.C].softmax(1).cpu().numpy())
    return np.concatenate(l1s), np.concatenate(ps)


def cube_of(name):
    if name in ("Jasper", "Urban4"):
        for n, cube, wl in h5.scenes():
            if n == name:
                return cube, wl, c.load(name)
    s = c.load(name)
    return s.cube.astype(np.float64), s.wl, s


def main():
    from sklearn.cluster import MiniBatchKMeans
    from app.spectral_match.namer.bank import build_bank
    from app.spectral_match.namer.infer import load_namer
    sys.path.insert(0, c.UPSTREAM)
    from engine.VCA import vca
    namer = load_namer(h5.N002, c.DEV)
    rng = np.random.default_rng(2)
    res = {}
    for name in c.DATASETS:
        cube, wl, s = cube_of(name)
        H, W, B = cube.shape
        bank = build_bank(wl, h5.CACHE, h5.LIB_ROOT)
        r = {"bands": int(B), "wl_range": [round(float(wl.min())), round(float(wl.max()))], "endmembers": []}
        # 1. endmember naming (GT in the dataset's endmember units; for Jasper / Urban4 rescaled to the cube's units)
        flat = cube.reshape(-1, B)
        A = s.A
        E_cube = np.linalg.lstsq(A.T, flat, rcond=None)[0] if name in ("Jasper", "Urban4") else s.E
        np.random.seed(0)
        Ev = np.clip(vca(flat.T, len(s.E), verbose=False)[0].T, 0, None)
        for tag, E in (("gt", E_cube), ("vca", Ev)):
            l1, p = namer_call(namer, E, E, wl, bank)
            for k in range(len(E)):
                em = s.em_names[k] if tag == "gt" else f"vca_{k + 1}"
                nearest = s.em_names[int(np.argmax(c.unit(E[k:k + 1]) @ c.unit(s.E).T))] if tag == "vca" else em
                r["endmembers"].append({"set": tag, "endmember": em, "nearest_gt": nearest, "label": label_of(name, nearest),
                                        "call": "man-made" if l1[k] > 0 else "natural",
                                        "p_manmade": round(float(1 / (1 + np.exp(-l1[k]))), 3),
                                        "class": CLASSES[int(p[k].argmax())], "p_class": round(float(p[k].max()), 3)})
        # 2. per-pixel naming
        ring, _, _ = h5.ring_and_noise(cube)
        yy, xx = np.mgrid[7:H - 7, 7:W - 7]
        rr, cc = yy.ravel(), xx.ravel()
        if len(rr) > 6000:
            k = rng.choice(len(rr), 6000, replace=False); rr, cc = rr[k], cc[k]
        Ak = A.reshape(len(s.E), H, W)[:, rr, cc]
        dom, amax = Ak.argmax(0), Ak.max(0)
        keep = amax >= 0.4
        rr, cc, dom, amax = rr[keep], cc[keep], dom[keep], amax[keep]
        lab = np.array([label_of(name, s.em_names[q]) for q in dom])
        X, R = cube[rr, cc], ring[rr, cc]
        Lr = bank.refl.numpy().astype(np.float64)
        lib_cls = np.where(bank.cls.numpy() < 0, h5.C, bank.cls.numpy())
        pc = lambda sc: np.stack([sc[:, lib_cls == q].max(1) for q in range(h5.C + 1)], 1)
        mu = flat.mean(0)
        cov = np.cov(flat[::3].T)
        Ci = np.linalg.inv(cov + 1e-6 * np.eye(B) * np.trace(cov) / B)
        cs = {"sam": pc(c.unit(X) @ c.unit(Lr).T)}
        for lvl, bh in (("ls_none", np.zeros_like(X)), ("ls_mean", np.repeat(mu[None], len(X), 0)), ("ls_ring", R)):
            cs[lvl] = pc(-h5.residual_scores(X, bh, Lr)[1])
        xm, tm = X - mu, Lr - mu
        num = (xm @ Ci) @ tm.T
        cs["ace"] = pc(np.sign(num) * num ** 2 / (np.einsum("pb,bc,pc->p", xm, Ci, xm)[:, None] * np.einsum("lb,bc,lc->l", tm, Ci, tm)[None]))
        protos = MiniBatchKMeans(12, random_state=0, n_init=3, batch_size=2048).fit(flat).cluster_centers_
        bank_sp = np.concatenate([(1 - a) * p_[None] + a * Lr for a in h5.BANK_A for p_ in protos])
        dist = torch.cdist(torch.as_tensor(X, device=c.DEV), torch.as_tensor(bank_sp, device=c.DEV)).cpu().numpy()
        cs["bank_refl"] = pc(-dist.reshape(len(X), -1, len(Lr)).min(1))
        f = c.Frame(flat)
        cs["lat_none"] = pc(c.unit(c.encode(f(X), wl)) @ c.unit(c.encode(f(Lr), wl)).T)
        out = {}
        for m, sc in cs.items():
            mm = sc[:, :h5.C]; top = np.sort(mm, 1)
            out[m] = (mm.max(1) - sc[:, h5.C], mm.argmax(1), top[:, -1] - top[:, -2])
        for tag, rm in (("namer_ring", R), ("namer_self", X)):
            l1, p = namer_call(namer, X, rm, wl, bank)
            out[tag] = (l1, p.argmax(1), 1 / (1 + np.exp(-l1)) * p.max(1))
        is_mm = np.isin(lab, ["concrete_asphalt", "manmade_l1"])
        is_nat = lab == "natural"
        pure = amax >= 0.9
        r["pixels"] = {"n": int(len(X)), "n_pure": int(pure.sum()), "n_mixed": int((~pure).sum()),
                       "n_by_label": {q: int((lab == q).sum()) for q in np.unique(lab)}, "methods": {}}
        for m, (l1, pred, conf) in out.items():
            rec = {}
            for st, sel in (("pure", pure), ("mixed", ~pure)):
                e = {}
                if (sel & is_mm).any() and (sel & is_nat).any():
                    y = is_mm[sel & (is_mm | is_nat)]
                    sc_ = l1[sel & (is_mm | is_nat)]
                    cut = np.quantile(l1[sel & is_nat], 0.95)
                    e["l1_auc"] = round(float(roc_auc_score(y, sc_)), 3)
                    e["l1_recall_5fa"] = round(float((l1[sel & is_mm] > cut).mean()), 3)
                if (sel & is_nat).any():
                    e["natural_called_manmade"] = round(float((l1[sel & is_nat] > 0).mean()), 3)   # others: best man-made > natural
                l2 = sel & (lab == "concrete_asphalt")
                if l2.any():
                    hit = pred[l2] == CLASSES.index("concrete_asphalt")
                    o = np.argsort(-conf[l2], kind="stable")
                    e["l2_acc"] = round(float(hit.mean()), 3)
                    e["l2_acc_cov50"] = round(float(hit[o[:max(len(o) // 2, 1)]].mean()), 3)
                    e["n_l2"] = int(l2.sum())
                rec[st] = e
            r["pixels"]["methods"][m] = rec
        res[name] = r
        print(name, json.dumps(r), flush=True)
    json.dump(res, open("results/h7_unmixing_sets_n002.json", "w"), indent=1)


if __name__ == "__main__":
    main()
