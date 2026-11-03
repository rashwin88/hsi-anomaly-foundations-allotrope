"""
Real measured targets (README "Real targets"): N002 (inference only, ring as in production) next to training-free
matchers on the same pixels and the same library.

Sets: Gudalur AVIRIS-NG 4 m (5 deployed sheets), the same coarsened to 32 m (8 x 8 block mean, 4 grid phases), DLR HySU
(HySpex VNIR 417-903 nm, 5 materials x 5 sizes). Truth: ROI pixels; background: pixels >= 10 px (HySU: 3 px) from any
target. Libraries: family_library resampled to the sensor bands (primary); "+field" adds the sets' own field spectra
(Gudalur point spectra, HySU SVC means) under their mapped class (secondary).

Level 1: man-made score (namer: level-1 logit; others: best man-made class score - natural class score) on target vs
background pixels -> AUC, recall at 5% background false alarms. Level 2: best man-made class on target pixels, strict
and lenient mapping (README table), accuracy at 100% / 50% coverage (confidence: namer P(man-made) x P(class), others the
top-1 minus top-2 man-made class score), and per-target majority vote.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import common as c
import h5_neighbourhood as h5

GUD = Path.home() / "hsu" / "gud" / "Data_in_brief_dataset"
HYSU = Path(c.DATA) / "hysu"
CLASSES = ("oil_fuel", "plastic", "textile_fabric", "wood_paper", "paint_coatings", "metal", "concrete_asphalt")
MAP = {"green_nylon_sheet": ("plastic", {"plastic", "textile_fabric"}), "red_nylon_sheet": ("plastic", {"plastic", "textile_fabric"}),
       "yellow_nylon_sheet": ("plastic", {"plastic", "textile_fabric"}), "black_nylon_sheet": ("plastic", {"plastic", "textile_fabric"}),
       "white_cotton_sheet": ("textile_fabric", {"textile_fabric"}),
       "bitumen": ("concrete_asphalt", {"concrete_asphalt", "oil_fuel"}), "red metal": ("paint_coatings", {"paint_coatings", "metal"}),
       "blue fabric": ("textile_fabric", {"textile_fabric", "plastic"}), "red fabric": ("textile_fabric", {"textile_fabric", "plastic"}),
       "green fabric": ("textile_fabric", {"textile_fabric", "plastic"})}


def _hdr_list(text, key):
    m = re.search(key + r"\s*=\s*\{([^}]*)\}", text, re.I)
    return np.array([float(v) for v in m.group(1).replace("\n", " ").split(",") if v.strip()]) if m else None


def gudalur():
    hdr = (GUD / "DATA/AIRBORNE/AVNG_PROCESSED/AVNG_REFL.hdr").read_text(errors="replace")
    H, W, B = (int(re.search(k + r"\s*=\s*(\d+)", hdr).group(1)) for k in ("lines", "samples", "bands"))
    wl = _hdr_list(hdr, "wavelength")
    bbl = _hdr_list(hdr, "bbl")
    raw = np.memmap(GUD / "DATA/AIRBORNE/AVNG_PROCESSED/AVNG_REFL", dtype="<i2", mode="r", shape=(H, B, W))
    masks, names, field = [], [], []
    for t in range(1, 6):
        txt = (GUD / f"TARGET_ROI/AVNG/T{t}/T{t}_A.txt").read_text(errors="replace")
        names.append(re.search(r"ROI name: T\d_A_(\S+)", txt).group(1).lower())
        xy = np.array([[int(v) for v in ln.split()[1:3]] for ln in txt.splitlines() if re.match(r"^\s*\d+\s+\d+\s+\d+", ln)])
        mk = np.zeros((H, W), bool); mk[xy[:, 1] - 1, xy[:, 0] - 1] = True
        masks.append(mk)
        sp = np.array([[float(v) for v in ln.split()] for ln in (GUD / f"TARGET_LIBRARY_SPECTRUM/PROCESSED_SPECTRUM/POINT_SPECTRUM/POINT_AVNG/ascii/T{t}.txt").read_text(errors="replace").splitlines() if re.match(r"^\s*[-0-9.]", ln)])
        field.append((sp[:, 0] * (1000 if sp[:, 0].max() < 10 else 1), sp[:, 1] / (100 if sp[:, 1].max() > 1.5 else 1)))
    allm = np.any(masks, 0)
    ys, xs = np.nonzero(allm)
    y0, y1, x0, x1 = max(ys.min() - 150, 0), min(ys.max() + 150, H), max(xs.min() - 150, 0), min(xs.max() + 150, W)
    cube = np.asarray(raw[y0:y1, :, x0:x1]).transpose(0, 2, 1).astype(np.float64) / 1e4
    keep = (bbl > 0) if bbl is not None else np.ones(B, bool)
    med = np.median(cube.reshape(-1, B)[::37], 0)
    keep &= (med > 0.005) & ~((wl > 1340) & (wl < 1450)) & ~((wl > 1790) & (wl < 1960)) & (wl >= 400) & (wl <= 2450)
    cube, wl = np.clip(cube[..., keep], 0, None), wl[keep]
    masks = [m[y0:y1, x0:x1] for m in masks]
    return cube, wl, masks, names, field


def coarsen(cube, masks, f=8, oy=0, ox=0):
    H, W, B = cube.shape
    h, w = (H - oy) // f, (W - ox) // f
    cc = cube[oy:oy + h * f, ox:ox + w * f].reshape(h, f, w, f, B).mean((1, 3))
    fills = [m[oy:oy + h * f, ox:ox + w * f].reshape(h, f, w, f).mean((1, 3)) for m in masks]
    return cc, fills


def hysu():
    import scipy.io as sio
    import tifffile
    hdr = (HYSU / "DLR_HySU_HS_full.hdr").read_text(errors="replace")
    wl = _hdr_list(hdr, "wavelength") * 1000
    cube = tifffile.imread(HYSU / "DLR_HySU_HS_full.tif").astype(np.float64) / 1e4
    m = sio.loadmat(HYSU / "masks.mat")
    mats = ["bitumen", "red metal", "blue fabric", "red fabric", "green fabric"]
    masks, names = [], []
    for k, n in enumerate(mats):
        for code in range(1, 6):
            mk = (m["materials"][..., k] > 0) & (m["targetsROI"] == code)
            if mk.any():
                masks.append(mk); names.append(n)
    rows = [ln for ln in (HYSU / "DLR_HySU_SVC_ASCII.txt").read_text(errors="replace").splitlines() if re.match(r"^\s*[-0-9.]", ln)]
    arr = np.array([[float(v) for v in r.split()] for r in rows])
    field = [(arr[:, 0] * 1000, arr[:, j + 1] / 1e4) for j in range(5)]
    return np.clip(cube, 0, None), wl, masks, names, field, mats


def collect(cube, wl, masks, names, field_spec, field_names, namer, rng, bg_gap=10, fill_maps=None):
    from scipy.ndimage import binary_dilation
    from sklearn.cluster import MiniBatchKMeans
    from app.spectral_match.namer.bank import build_bank, LibraryBank
    H, W, B = cube.shape
    ring, _, _ = h5.ring_and_noise(cube)
    tgt = np.zeros((H, W), int)
    for i, mk in enumerate(masks):
        tgt[mk & (tgt == 0)] = i + 1
    bgm = ~binary_dilation(tgt > 0, iterations=bg_gap)
    bgm[:7], bgm[-7:], bgm[:, :7], bgm[:, -7:] = False, False, False, False
    br, bc = np.nonzero(bgm)
    k = rng.choice(len(br), min(3000, len(br)), replace=False)
    tr, tcol = np.nonzero(tgt > 0)
    rr, cc = np.concatenate([tr, br[k]]), np.concatenate([tcol, bc[k]])
    tid = tgt[rr, cc]
    X, R = cube[rr, cc], ring[rr, cc]
    flat = cube.reshape(-1, B)
    mu, cov = flat.mean(0), np.cov(flat[::7].T)
    Ci = np.linalg.inv(cov + 1e-6 * np.eye(B) * np.trace(cov) / B)
    f = c.Frame(flat)
    protos = MiniBatchKMeans(12, random_state=0, n_init=3, batch_size=2048).fit(cube[bgm]).cluster_centers_
    out = {"n_target_px": int(len(tr)), "n_background_px": int(len(k)),
           "bands": int(B), "wl_range": [round(float(wl.min())), round(float(wl.max()))], "libraries": {}}
    base = build_bank(wl, h5.CACHE, h5.LIB_ROOT)
    for lib_tag in ("family", "+field"):
        bank = base
        if lib_tag == "+field":
            add = np.stack([np.interp(wl, w_, s_) for w_, s_ in field_spec])
            cl = torch.tensor([CLASSES.index(MAP[n][0]) for n in field_names])
            bank = LibraryBank(torch.cat([base.refl, torch.as_tensor(add, dtype=torch.float32)]), torch.cat([base.cls, cl]),
                               torch.cat([base.test, torch.zeros(len(add), dtype=torch.bool)]), base.names + field_names)
        Lr = bank.refl.numpy().astype(np.float64)
        lib_cls = np.where(bank.cls.numpy() < 0, h5.C, bank.cls.numpy())
        pc = lambda s: np.stack([s[:, lib_cls == q].max(1) for q in range(h5.C + 1)], 1)
        cs = {"sam": pc(c.unit(X) @ c.unit(Lr).T),
              "ls_ring": pc(-h5.residual_scores(X, R, Lr)[1]),
              "ls_mean": pc(-h5.residual_scores(X, np.repeat(mu[None], len(X), 0), Lr)[1])}
        xm, tm = X - mu, Lr - mu
        num = (xm @ Ci) @ tm.T
        cs["ace"] = pc(np.sign(num) * num ** 2 / (np.einsum("pb,bc,pc->p", xm, Ci, xm)[:, None] * np.einsum("lb,bc,lc->l", tm, Ci, tm)[None]))
        bank_sp = np.concatenate([(1 - a) * p[None] + a * Lr for a in h5.BANK_A for p in protos])
        dist = torch.cdist(torch.as_tensor(X, device=c.DEV), torch.as_tensor(bank_sp, device=c.DEV)).cpu().numpy()
        cs["bank_refl"] = pc(-dist.reshape(len(X), -1, len(Lr)).min(1))
        zL = c.unit(c.encode(f(Lr), wl))
        zX = c.unit(c.encode(f(X), wl))
        cs["lat_none"] = pc(zX @ zL.T)
        zb = c.unit(c.encode(f(bank_sp), wl))
        cs["bank_lat"] = pc((zX @ zb.T).reshape(len(X), -1, len(Lr)).max(1))
        res = {}
        for m, s in cs.items():
            mm = s[:, :h5.C]
            top = np.sort(mm, 1)
            res[m] = (mm.max(1) - s[:, h5.C], mm.argmax(1), top[:, -1] - top[:, -2])
        dev = c.DEV
        w = torch.as_tensor(wl, dtype=torch.float32, device=dev)
        with torch.no_grad():
            lib = namer.forward_library(bank.refl.to(dev), w)
            lc = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, h5.C)).to(dev)
            for ntag, rm in (("namer_ring", R), ("namer_self", X)):
                z, _, _ = namer.forward_pixel(torch.as_tensor(X, dtype=torch.float32, device=dev), torch.as_tensor(rm, dtype=torch.float32, device=dev), w)
                sc = namer.class_scores(z, lib, lc, h5.C + 1)
                l1 = namer.level1_logit(sc)
                p = sc[:, :h5.C].softmax(1)
                res[ntag] = (l1.cpu().numpy(), p.argmax(1).cpu().numpy(), (torch.sigmoid(l1) * p.max(1).values).cpu().numpy())
        out["libraries"][lib_tag] = res
    out["tid"], out["names"] = tid, names
    out["fill"] = np.max([fm[rr, cc] for fm in fill_maps], 0) if fill_maps is not None else None
    return out


def score(tag, raws):
    """Metrics over one or more pooled collect() outputs (target ids offset per raw so targets stay distinct)."""
    names, tid, fills, off = [], [], [], 0
    for r in raws:
        names += r["names"]; tid.append(np.where(r["tid"] > 0, r["tid"] + off, 0)); off += len(r["names"])
        fills.append(r["fill"] if r["fill"] is not None else np.full(len(r["tid"]), np.nan))
    tid, fills = np.concatenate(tid), np.concatenate(fills)
    out = {k: (sum(r[k] for r in raws) if k.startswith("n_") else raws[0][k]) for k in raws[0] if k.startswith(("n_", "bands", "wl_"))}
    out["n_targets"] = int(len(np.unique(tid[tid > 0])))
    out["libraries"] = {}
    is_t = tid > 0
    strict = np.array([CLASSES.index(MAP[names[t - 1]][0]) if t else -1 for t in tid])
    lenient = [MAP[names[t - 1]][1] if t else set() for t in tid]
    for lib_tag in raws[0]["libraries"]:
        tab = {}
        for m in raws[0]["libraries"][lib_tag]:
            l1, pred, conf = (np.concatenate([r["libraries"][lib_tag][m][i] for r in raws]) for i in range(3))
            cut = np.quantile(l1[~is_t], 0.95)
            hit_s = (pred == strict)[is_t]
            hit_l = np.array([CLASSES[p_] in L for p_, L in zip(pred, lenient)])[is_t]
            o = np.argsort(-conf[is_t], kind="stable")
            half = o[:max(len(o) // 2, 1)]
            votes = []
            for t in np.unique(tid[is_t]):
                sel = tid[is_t] == t
                maj = np.bincount(pred[is_t][sel], minlength=h5.C).argmax()
                votes.append(CLASSES[maj] in MAP[names[t - 1]][1])
            tab[m] = {"l1_auc": round(float(roc_auc_score(is_t, l1)), 3), "l1_recall_5fa": round(float((l1[is_t] > cut).mean()), 3),
                      "l2_strict": round(float(hit_s.mean()), 3), "l2_lenient": round(float(hit_l.mean()), 3),
                      "l2_lenient_cov50": round(float(hit_l[half].mean()), 3),
                      "targets_majority_lenient": f"{int(np.sum(votes))}/{len(votes)}"}
            fv = fills[is_t]
            if np.isfinite(fv).any():
                tab[m]["l1_recall_5fa_by_fill"] = {f"{lo}-{hi}": round(float((l1[is_t][(fv >= lo) & (fv < hi)] > cut).mean()), 3)
                                                   for lo, hi in ((0.05, 0.25), (0.25, 0.5), (0.5, 1.01)) if ((fv >= lo) & (fv < hi)).any()}
        out["libraries"][lib_tag] = tab
    print(tag, json.dumps(out), flush=True)
    return out


def main():
    from app.spectral_match.namer.infer import load_namer
    namer = load_namer(h5.N002, c.DEV)
    rng = np.random.default_rng(0)
    res = {}
    cube, wl, masks, names, field = gudalur()
    res["gudalur_4m"] = score("gudalur_4m", [collect(cube, wl, masks, names, field, names, namer, rng)])
    res["gudalur_4m"]["targets"] = {n: int(m.sum()) for n, m in zip(names, masks)}
    raws = []                                   # 32 m: 4 grid phases pooled; target pixel = fill >= 0.05
    for oy, ox in ((0, 0), (0, 4), (4, 0), (4, 4)):
        cc_, fl = coarsen(cube, masks, 8, oy, ox)
        raws.append(collect(cc_, wl, [fi >= 0.05 for fi in fl], names, field, names, namer, rng, bg_gap=2, fill_maps=fl))
    res["gudalur_32m"] = score("gudalur_32m", raws)
    cube, wl, masks, names, field, mats = hysu()
    res["dlr_hysu"] = score("dlr_hysu", [collect(cube, wl, masks, names, field, mats, namer, rng, bg_gap=3)])
    res["dlr_hysu"]["targets"] = [(n, int(m.sum())) for n, m in zip(names, masks)]
    json.dump(res, open("results/h6_real_targets.json", "w"), indent=1)


if __name__ == "__main__":
    main()
