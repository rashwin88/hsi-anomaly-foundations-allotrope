"""
H5 - does the neighbourhood matter for matching? (README, the headline experiment)

Planted single-pixel targets x = (1 - a) b + a t + n into real backgrounds b (Jasper, Urban4, 8 PRISMA test tiles),
t a man-made family-library spectrum (namer v1 classes), a on a fine grid, n ~ N(0, sigma_band) with sigma from
horizontal neighbour differences. Task: name t's class (7 man-made classes + "natural" = 8-way), its own spectrum
left out of the library (class level, as the namer is scored).

Background estimates b^:  none (0) | scene mean mu | ring mean (3-7 px Chebyshev ring, the namer's)
Reflectance arms:
    ls_<lvl>        per-candidate least squares: a_j = (t_j - b^).(x - b^) / |t_j - b^|^2 clipped to [0, 1],
                    r_j = |x - b^ - a_j (t_j - b^)|, class of argmin r_j
    fix_<lvl>_<k>   fixed a^ = k * a (k = 1 oracle, 0.85): t^ = (x - (1 - a^) b^) / a^, SAM to the library
    ace             ACE with the scene mean and covariance (whitened scene-global)
    bank_refl       scene k-means prototypes (12) x library x a in BANK_A, nearest mixture (Euclidean)
Latent arms (frozen HyperSL, scene frame):
    lat_none        cos(z(x), z(t_j))
    lat_<lvl>       t^ from ls_<lvl>'s best a, then cos(z(t^), z(t_j))
    lat_ring_or     t^ with the oracle a
    bank_lat        the prototype x library x a bank, matched by cosine in latent space
N002 (inference only): namer_ring (ring as supplied) | namer_self (ring := the pixel, context neutralised).

Theory (reflectance, per sample): correct when a d > (1 - a) |delta_perp| + |n_perp|, delta = b - b^, perps taken
against (t - b^), d = the class margin of the pure noise-free target (t - b^) in the same residual metric (min
wrong-class residual - min same-class residual). Predicted accuracy(a) = share of samples with a > a*_i.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

import common as c

ALPHAS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
BANK_A = [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
REPO = Path(__file__).resolve().parents[3]
LIB_ROOT = REPO / "data" / "family_library"
CACHE = Path.home() / "hsu" / "lib_cache"
N002 = Path.home() / "hsu" / "N002" / "last.pt"
C = 7                                                             # man-made classes; natural = 7
CONF: list = []                                                   # last class_pick margin
PER_SCENE = {"Jasper": 1000, "Urban4": 1000, "prisma": 150}       # prisma: per tile


def scenes():
    d = sio.loadmat(f"{c.DATA}/Jasper.mat")                           # AVIRIS, x 1e4 reflectance
    H, W = int(d["lines"].squeeze()), int(d["cols"].squeeze())
    Y = d["Y"].astype(np.float64).T.reshape(W, H, -1).transpose(1, 0, 2) / 1e4
    yield "Jasper", Y, np.linspace(380, 2500, 224)[c._drop(224, [(1, 3), (108, 112), (154, 166), (220, 224)])]
    d = sio.loadmat(f"{c.DATA}/Urban4.mat")                           # HYDICE, x 1e3 reflectance
    H, W = int(d["lines"].squeeze()), int(d["cols"].squeeze())
    Y = d["Y"].astype(np.float64).T.reshape(W, H, -1).transpose(1, 0, 2) / 1e3
    yield "Urban4", Y, np.linspace(400, 2500, 210)[c._drop(210, [(1, 4), (76, 76), (87, 87), (101, 111), (136, 153), (198, 210)])]
    z = np.load(Path.home() / "hsu" / "prisma_tiles.npz")
    for k in sorted({k.split("__")[0] for k in z.files}):
        raw, wl = z[f"{k}__pix"], z[f"{k}__wl"].astype(np.float64)
        has = (raw != -32768).any(0)                                  # pixels inside the swath
        live = (raw[:, has] != -32768).mean(1) > 0.99                 # drop bands dead at > 1% of them (water)
        o = np.argsort(wl)
        o = o[live[o]]
        cube = np.where(raw == -32768, np.nan, raw * 1e-4)[o].transpose(1, 2, 0)
        yield f"prisma:{k}", cube, wl[o]


def ring_and_noise(cube):
    """ring mean (H, W, B) over the 3-7 px Chebyshev ring (15x15 box minus 5x5), ring CV (H, W), sigma (B,)."""
    from scipy.ndimage import uniform_filter
    H, W, B = cube.shape
    ring = np.empty_like(cube)
    cv = np.zeros((H, W))
    for b in range(B):
        v = cube[..., b]
        s1 = uniform_filter(v, 15, mode="constant") * 225 - uniform_filter(v, 5, mode="constant") * 25
        s2 = uniform_filter(v * v, 15, mode="constant") * 225 - uniform_filter(v * v, 5, mode="constant") * 25
        ring[..., b] = s1 / 200
        cv += np.sqrt(np.maximum(s2 / 200 - (s1 / 200) ** 2, 0)) / np.maximum(np.abs(s1 / 200), 1e-6)
    sigma = np.sqrt((np.diff(cube, axis=1) ** 2).mean(axis=(0, 1))) / np.sqrt(2)
    return ring, cv / B, sigma


def residual_scores(X, Bh, L):
    """X (P, B), Bh (P, B) background estimates, L (Lib, B) -> (alpha (P, Lib) clipped, residual (P, Lib))."""
    X, Bh, L = (torch.as_tensor(v, dtype=torch.float64, device=c.DEV) for v in (X, Bh, L))
    Yv = X - Bh
    yv = (Yv * Yv).sum(1, keepdim=True)
    yl = Yv @ L.T - (Yv * Bh).sum(1, keepdim=True)                    # <y, t_j - b^>
    vv = (L * L).sum(1)[None] - 2 * Bh @ L.T + (Bh * Bh).sum(1, keepdim=True)
    a = (yl / vv.clamp(min=1e-12)).clamp(0, 1)
    r2 = yv - 2 * a * yl + a * a * vv
    return a.cpu().numpy(), np.sqrt(np.maximum(r2.cpu().numpy(), 0))


def class_pick(score, lib_cls, own, higher=True):
    """score (P, Lib); own (P,) entry left out -> (pred (P,) in 0..7, class scores (P, 8))."""
    s = np.where(higher, score, -score).astype(np.float64).copy()
    s[np.arange(len(s)), own] = -np.inf
    cs = np.stack([s[:, lib_cls == k].max(1) for k in range(C + 1)], 1)
    top = np.sort(cs, 1)
    CONF.append(top[:, -1] - top[:, -2])                           # confidence: top-1 minus top-2 class score
    return cs.argmax(1), cs


def margin_d(T, Bh, Lref, lib_cls, own, cls):
    """Class margin of the pure, noise-free target in the residual metric."""
    _, r = residual_scores(T, Bh, Lref)
    r[np.arange(len(r)), own] = np.inf
    same = np.where(lib_cls[None] == cls[:, None], r, np.inf).min(1)
    wrong = np.where(lib_cls[None] != cls[:, None], r, np.inf).min(1)
    return wrong - same


def perp(v, u):
    u = c.unit(u)
    return np.linalg.norm(v - (v * u).sum(1, keepdims=True) * u, axis=1)


def run_scene(name, cube, wl, rng, namer):
    from app.spectral_match.namer.bank import build_bank
    from sklearn.cluster import MiniBatchKMeans
    H, W, B = cube.shape
    ok = np.isfinite(cube).all(2)
    ring, cv, sigma = ring_and_noise(np.nan_to_num(cube))
    ok[:7], ok[-7:], ok[:, :7], ok[:, -7:] = False, False, False, False
    from scipy.ndimage import minimum_filter
    ok &= minimum_filter(np.isfinite(cube).all(2).astype(np.uint8), 15) > 0
    rr, cc = np.nonzero(ok)
    if len(rr) < 50:                                              # a tile with no fully valid 15 x 15 window
        print(name, "skipped: too few valid pixels", len(rr), flush=True)
        return None
    n = PER_SCENE["prisma" if name.startswith("prisma") else name]
    pick = rng.choice(len(rr), min(n, len(rr)), replace=False)
    rr, cc = rr[pick], cc[pick]
    P = len(rr)
    bank = build_bank(wl, CACHE, LIB_ROOT)
    Lr = bank.refl.numpy().astype(np.float64)
    lib_cls = np.where(bank.cls.numpy() < 0, C, bank.cls.numpy())
    mm = np.flatnonzero(lib_cls < C)
    own = mm[rng.integers(0, len(mm), P)]
    cls = lib_cls[own]
    T = Lr[own]
    b = cube[rr, cc]
    bring = ring[rr, cc]
    flat = cube[ok]
    mu = flat.mean(0)
    cov = np.cov(flat.T) + 1e-6 * np.eye(B) * np.trace(np.cov(flat.T)) / B
    Ci = np.linalg.inv(cov)
    noise = rng.standard_normal((P, B)) * sigma
    f = c.Frame(np.nan_to_num(flat))
    zL = c.encode(f(Lr), wl)
    protos = MiniBatchKMeans(12, random_state=0, n_init=3, batch_size=2048).fit(flat).cluster_centers_
    bank_sp = np.concatenate([(1 - a) * p[None] + a * Lr for a in BANK_A for p in protos])      # (A*12*Lib, B)
    z_bank = c.encode(f(bank_sp), wl)
    est = {"none": np.zeros_like(b), "mean": np.repeat(mu[None], P, 0), "ring": bring}
    delta = {k: b - v for k, v in est.items()}
    rec = {"scene": name, "P": P, "cls": cls.tolist(), "own": own.tolist(), "ring_bias": (np.linalg.norm(delta["ring"], axis=1) / np.linalg.norm(b, axis=1)).tolist(),
           "ring_cv": cv[rr, cc].tolist(), "pred": {}, "theory": {}}
    for k in ("none", "mean", "ring"):
        d = margin_d(T, est[k], Lr, lib_cls, own, cls)
        dp = perp(delta[k], T - est[k])
        npp = perp(noise, T - est[k])
        rec["theory"][k] = {"d": d.tolist(), "delta_perp": dp.tolist(), "n_perp": npp.tolist(),
                            "a_star": np.where(d > 0, (dp + npp) / np.maximum(d + dp, 1e-12), np.inf).tolist()}
    torch_dev = c.DEV
    for a in ALPHAS:
        X = (1 - a) * b + a * T + noise
        out = Rec()
        for k in ("none", "mean", "ring"):
            al, r = residual_scores(X, est[k], Lr)
            out[f"ls_{k}"] = class_pick(r, lib_cls, own, higher=False)[0]
            best_a = np.maximum(al[np.arange(P), np.where(np.arange(len(Lr))[None] == own[:, None], np.inf, r).argmin(1)], 0.01)
            if k != "none":
                for kk in (1.0, 0.85):
                    th = (X - (1 - kk * a) * est[k]) / (kk * a)
                    out[f"fix_{k}_{kk}"] = class_pick(c.unit(th) @ c.unit(Lr).T, lib_cls, own)[0]
            th = np.clip((X - (1 - best_a[:, None]) * est[k]) / best_a[:, None], 0, 1.5)
            zt = c.encode(f(th), wl)
            out[f"lat_{k}"] = class_pick(c.unit(zt) @ c.unit(zL).T, lib_cls, own)[0]
        th = np.clip((X - (1 - a) * bring) / a, 0, 1.5)
        out["lat_ring_or"] = class_pick(c.unit(c.encode(f(th), wl)) @ c.unit(zL).T, lib_cls, own)[0]
        xm, tm = X - mu, Lr - mu
        num = (xm @ Ci) @ tm.T
        ace = np.sign(num) * num ** 2 / (np.einsum("pb,bc,pc->p", xm, Ci, xm)[:, None] * np.einsum("lb,bc,lc->l", tm, Ci, tm)[None])
        out["ace"] = class_pick(ace, lib_cls, own)[0]
        dist = torch.cdist(torch.as_tensor(X, device=torch_dev), torch.as_tensor(bank_sp, device=torch_dev)).cpu().numpy()
        zx = c.encode(f(X), wl)
        cosb = c.unit(zx) @ c.unit(z_bank).T
        ent_best_d = dist.reshape(P, -1, len(Lr)).min(1)                # bank rows: [a][prototype][entry]
        ent_best_c = cosb.reshape(P, -1, len(Lr)).max(1)
        out["bank_refl"] = class_pick(ent_best_d, lib_cls, own, higher=False)[0]
        out["bank_lat"] = class_pick(ent_best_c, lib_cls, own)[0]
        out["lat_none"] = class_pick(c.unit(zx) @ c.unit(zL).T, lib_cls, own)[0]
        if namer is not None:
            for tag, rmean in (("namer_ring", bring), ("namer_self", X)):
                out[tag] = namer_pred(namer, X, rmean, wl, bank, own)
        rec["pred"][str(a)] = {k: v.tolist() for k, v in out.items()}
        rec.setdefault("conf", {})[str(a)] = {k: np.round(v, 5).tolist() for k, v in out.conf.items()}
        acc = {k: round(float((v == cls).mean()), 3) for k, v in out.items()}
        print(name, a, json.dumps(acc), flush=True)
    return rec


class Rec(dict):
    """dict of predictions that also keeps each arm's confidence (the CONF margin pushed by class_pick, or the
    namer's own)."""
    def __init__(self):
        super().__init__(); self.conf = {}

    def __setitem__(self, k, v):
        if isinstance(v, tuple):
            v, self.conf[k] = v
        else:
            self.conf[k] = CONF[-1]
        super().__setitem__(k, v)


@torch.no_grad()
def namer_pred(model, X, R, wl, bank, own):
    dev = next(model.parameters()).device
    w = torch.as_tensor(wl, dtype=torch.float32, device=dev)
    lib = model.forward_library(bank.refl.to(dev), w)
    lib_cls = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, C)).to(dev)
    preds, conf = [], []
    for i in range(0, len(X), 4096):
        z, _, _ = model.forward_pixel(torch.as_tensor(X[i:i + 4096], dtype=torch.float32, device=dev),
                                      torch.as_tensor(R[i:i + 4096], dtype=torch.float32, device=dev), w)
        sc = model.class_scores(z, lib, lib_cls, C + 1, exclude=torch.as_tensor(own[i:i + 4096], device=dev))
        l1 = model.level1_logit(sc)
        preds.append(torch.where(l1 > 0, sc[:, :C].argmax(1), torch.full_like(l1, C, dtype=torch.long)).cpu().numpy())
        conf.append((torch.sigmoid(l1) * sc[:, :C].softmax(1).max(1).values).cpu().numpy())   # P(named class)
    return np.concatenate(preds), np.concatenate(conf)


def main():
    from app.spectral_match.namer.infer import load_namer
    namer = load_namer(N002, c.DEV) if N002.exists() else None
    rng = np.random.default_rng(0)
    only = sys.argv[1:] or None
    recs = []
    if only and Path("results/h5_raw.json").exists():               # re-run some scenes, keep the others
        recs = [r for r in json.load(open("results/h5_raw.json")) if not any(r["scene"].startswith(o) for o in only)]
    for name, cube, wl in scenes():
        if only and not any(name.startswith(o) for o in only):
            continue
        r = run_scene(name, cube, wl, rng, namer)
        if r is not None:
            recs.append(r)
        json.dump(recs, open("results/h5_raw.json", "w"))


if __name__ == "__main__":
    main()
