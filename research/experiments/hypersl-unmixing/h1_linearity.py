"""
H1 - does the frozen HyperSL encoder keep linear mixing linear, within one scene's normalisation frame (README)?

For every pair of a scene's ground-truth endmembers (e0, e1): mixtures m(a) = (1 - a) e0 + a e1 in the scene frame,
pooled embedding z(a). Per pair:
    dev_max   max_a |z(a) - (z0 + a c)| / |c|,  c = z1 - z0          (chord deviation, normalised)
    t(a)      <z(a) - z0, c> / |c|^2                                  (position along the chord; linear means t = a)
    mono      share of grid steps where t increases; t_err = max |t(a) - a|
    slope     log-log slope of the tangent residual |z(a) - z0 - a g| over a in [0.01, 0.2], g the tangent at 0
              (2 = smooth, quadratic departure; 1 = the first-order term is already wrong)
    lin_range ratio |z(a) - z0| / (a |g|) at a = 0.05, 0.2, 0.5 (1 = still first order)
For triples: barycentric grid (step 0.1), deviation from the plane of the three corner embeddings and the error of
the barycentric weights recovered by FCLS in latent space (corners known).
"""

from __future__ import annotations

import itertools
import json

import numpy as np

import common as c

GRID = np.round(np.arange(0, 1.0001, 0.05), 4)
SMALL = np.array([0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2])


def pair_metrics(z0, z1, zg, zs, g):
    cvec = z1 - z0
    n2 = float(cvec @ cvec)
    dev = np.linalg.norm(zg - (z0 + GRID[:, None] * cvec), axis=1) / np.sqrt(n2)
    t = (zg - z0) @ cvec / n2
    res = np.linalg.norm(zs - z0 - SMALL[:, None] * g, axis=1)
    sel = (SMALL >= 0.01) & (SMALL <= 0.2)
    slope = float(np.polyfit(np.log(SMALL[sel]), np.log(np.maximum(res[sel], 1e-12)), 1)[0])
    gn = np.linalg.norm(g)
    lin = {f"lin_{a}": float(np.linalg.norm(zg[np.argmin(np.abs(GRID - a))] - z0) / (a * gn)) for a in (0.05, 0.2, 0.5)}
    return dict(dev_max=float(dev.max()), dev_mean=float(dev.mean()), mono=float((np.diff(t) > 0).mean()),
                t_err=float(np.abs(t - GRID).max()), slope=slope, **lin), dev, t


def main():
    rows, curves, tri_rows = [], {}, []
    rng = np.random.default_rng(0)
    for name in c.H_DATASETS:
        s = c.load(name)
        f = c.Frame(s.Y)
        K = len(s.E)
        curves[name] = {"dev": [], "t": []}
        for i, j in itertools.combinations(range(K), 2):
            a_all = np.concatenate([GRID, SMALL])
            mix = (1 - a_all[:, None]) * s.E[i] + a_all[:, None] * s.E[j]
            z = c.encode(f(mix), s.wl)
            zg, zs = z[:len(GRID)], z[len(GRID):]
            g = (zs[0] - zg[0]) / SMALL[0]
            m, dev, t = pair_metrics(zg[0], zg[-1], zg, zs, g)
            rows.append(dict(dataset=name, pair=f"{s.em_names[i]}-{s.em_names[j]}", **m,
                             sam_deg=float(np.degrees(np.arccos(np.clip(c.unit(s.E[i]) @ c.unit(s.E[j]), -1, 1))))))
            curves[name]["dev"].append(dev.tolist()); curves[name]["t"].append(t.tolist())
        triples = list(itertools.combinations(range(K), 3))
        if len(triples) > 20:
            triples = [triples[k] for k in rng.choice(len(triples), 20, replace=False)]
        W = np.array([(a, b, 10 - a - b) for a in range(11) for b in range(11 - a)], float) / 10
        for tr in triples:
            Et = s.E[list(tr)]
            z = c.encode(f(W @ Et), s.wl)
            corners = z[[np.flatnonzero(W[:, k] == 1)[0] for k in range(3)]]
            edge = np.mean([np.linalg.norm(corners[a] - corners[b]) for a, b in ((0, 1), (0, 2), (1, 2))])
            dev = np.linalg.norm(z - W @ corners, axis=1) / edge
            w_hat = c.fcls(corners @ corners.T, z @ corners.T)
            w_ref = c.fcls(f(Et) @ f(Et).T.astype(np.float64), f(W @ Et).astype(np.float64) @ f(Et).T)
            tri_rows.append(dict(dataset=name, triple="-".join(s.em_names[k] for k in tr), dev_max=float(dev.max()),
                                 dev_mean=float(dev.mean()), w_err_latent=float(np.abs(w_hat - W).mean()),
                                 w_err_frame=float(np.abs(w_ref - W).mean())))
        print(name, "pairs", sum(r["dataset"] == name for r in rows), "triples", len(triples), flush=True)
    summ = {}
    for name in c.H_DATASETS:
        pr = [r for r in rows if r["dataset"] == name]
        tr = [r for r in tri_rows if r["dataset"] == name]
        q = lambda key, rr: [round(float(np.percentile([r[key] for r in rr], p)), 3) for p in (50, 90)]
        summ[name] = dict(n_pairs=len(pr), dev_max_med_p90=q("dev_max", pr), t_err_med_p90=q("t_err", pr),
                          mono_min=round(min(r["mono"] for r in pr), 3), slope_med=q("slope", pr)[0],
                          lin_005_med=q("lin_0.05", pr)[0], lin_02_med=q("lin_0.2", pr)[0], lin_05_med=q("lin_0.5", pr)[0],
                          n_triples=len(tr), tri_dev_max_med=q("dev_max", tr)[0] if tr else None,
                          tri_w_err_latent_med=q("w_err_latent", tr)[0] if tr else None,
                          tri_w_err_frame_med=q("w_err_frame", tr)[0] if tr else None)
    json.dump(dict(summary=summ, pairs=rows, triples=tri_rows, grid=GRID.tolist(), curves=curves),
              open("results/h1.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    plot(curves)


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, cv in curves.items():
        d, t = np.array(cv["dev"]), np.array(cv["t"])
        ax[0].plot(GRID, np.median(d, 0), label=name)
        ax[1].plot(GRID, np.median(t, 0), label=name)
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="mixing fraction a (scene frame)", ylabel="|z(a) - chord| / |z1 - z0| (median over pairs)",
              title="H1: deviation from the chord")
    ax[1].set(xlabel="mixing fraction a", ylabel="projection t(a) on the chord (median)", title="H1: position on chord")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/h1_linearity.png", dpi=130)


if __name__ == "__main__":
    main()
