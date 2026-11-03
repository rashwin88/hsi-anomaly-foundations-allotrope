"""
H2 - FCLS in HyperSL latent space vs FCLS in reflectance, with the SAME endmembers (README).

Endmember sets: ground truth, and VCA (upstream engine/VCA.py, seed 0) aligned to ground truth by Hungarian on SAD.
Spaces (all FCLS, the same batched solver):
    refl        raw reflectance
    frame       scene-frame reflectance (per-band min-max; the space HyperSL sees)
    latent      pooled HyperSL embedding (global token + mean of band tokens)
    tokens      all band tokens concatenated (B x 256 per pixel), via Gram sums
Metrics: aRMSE (RMSE over all abundances), AAD (mean angle between estimated and true abundance vectors, deg).
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import common as c


def scores(A_hat, A):
    rmse = float(np.sqrt(((A_hat - A) ** 2).mean()))
    cos = (c.unit(A_hat.T) * c.unit(A.T)).sum(1)
    return {"aRMSE": round(rmse, 4), "AAD_deg": round(float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean()), 2)}


def unmix_all(s, E, f):
    out = {}
    out["refl"] = c.fcls(E @ E.T, s.Y @ E.T).T
    Ef, Yf = f(E).astype(np.float64), f(s.Y).astype(np.float64)
    out["frame"] = c.fcls(Ef @ Ef.T, Yf @ Ef.T).T
    tokE = {}
    zE = c.encode(f(E), s.wl, tokens_fn=lambda i, t: tokE.__setitem__("t", t.double()))
    TE = tokE["t"]                                                          # (K, B, 256)
    bt = []
    zY = c.encode(f(s.Y), s.wl, batch=512,
                  tokens_fn=lambda i, t: bt.append(torch.einsum("nbd,kbd->nk", t.double(), TE).cpu().numpy()))
    out["latent"] = c.fcls(zE @ zE.T, zY @ zE.T).T
    out["tokens"] = c.fcls(torch.einsum("kbd,lbd->kl", TE, TE).cpu().numpy(), np.concatenate(bt)).T
    return out


def main():
    sys.path.insert(0, c.UPSTREAM)
    from engine.VCA import vca
    res = {}
    for name in c.H_DATASETS:
        s = c.load(name)
        f = c.Frame(s.Y)
        K = len(s.E)
        np.random.seed(0)
        Ev = np.clip(vca(s.Y.T, K, verbose=False)[0].T, 0, None)              # (K, B)
        sad = np.degrees(np.arccos(np.clip(c.unit(Ev) @ c.unit(s.E).T, -1, 1)))
        r, cc = linear_sum_assignment(sad)
        Ev_al = np.empty_like(Ev); Ev_al[cc] = Ev[r]
        res[name] = {"N": int(len(s.Y)), "K": K, "B": int(s.Y.shape[1]), "vca_mSAD_deg": round(float(sad[r, cc].mean()), 2)}
        for label, E in (("gt", s.E), ("vca", Ev_al)):
            A_hat = unmix_all(s, E, f)
            res[name][label] = {sp: scores(a, s.A) for sp, a in A_hat.items()}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open("results/h2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
