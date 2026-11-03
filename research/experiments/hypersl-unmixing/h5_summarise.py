"""H5 summary: accuracy vs fill per arm (spectrum-level cluster bootstrap), theory vs empirical, heterogeneity strata,
ring contamination under a PSF. Reads results/h5_raw.json, writes results/h5_summary.json and figures/h5_*.png."""

from __future__ import annotations

import json

import numpy as np

ARMS = ["ls_none", "ls_mean", "ls_ring", "fix_mean_1.0", "fix_mean_0.85", "fix_ring_1.0", "fix_ring_0.85", "ace",
        "bank_refl", "lat_none", "lat_mean", "lat_ring", "lat_ring_or", "bank_lat", "namer_ring", "namer_self"]
TABLE_A = ["0.05", "0.1", "0.2", "0.5", "1.0"]


def groups(recs):
    g = {}
    for r in recs:
        key = "PRISMA" if r["scene"].startswith("prisma") else r["scene"]
        g.setdefault(key, []).append(r)
    out = {}
    for key, rs in g.items():
        cat = lambda f: np.concatenate([np.asarray(f(r)) for r in rs])
        d = {"cls": cat(lambda r: r["cls"]), "own": cat(lambda r: r["own"]), "ring_bias": cat(lambda r: r["ring_bias"]),
             "ring_cv": cat(lambda r: r["ring_cv"]), "pred": {}, "theory": {}}
        for a in rs[0]["pred"]:
            d["pred"][a] = {m: cat(lambda r: r["pred"][a][m]) for m in rs[0]["pred"][a]}
        for lvl in rs[0]["theory"]:
            d["theory"][lvl] = {k: cat(lambda r: r["theory"][lvl][k]) for k in rs[0]["theory"][lvl]}
        d["conf"] = {a: {m: cat(lambda r: r["conf"][a][m]) for m in rs[0]["conf"][a]} for a in rs[0].get("conf", {})}
        d["n_scenes"] = len(rs)
        out[key] = d
    return out


def boot(correct, own, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.unique(own)
    idx = [np.flatnonzero(own == e) for e in ids]
    vals = [correct[np.concatenate([idx[i] for i in rng.integers(0, len(ids), len(ids))])].mean() for _ in range(reps)]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def a_half(alphas, acc):
    """Fill where accuracy first reaches half of its plateau (accuracy at fill 1), linear interpolation."""
    target = acc[-1] / 2
    for i in range(1, len(acc)):
        if acc[i] >= target > acc[i - 1]:
            return round(float(alphas[i - 1] + (target - acc[i - 1]) / (acc[i] - acc[i - 1]) * (alphas[i] - alphas[i - 1])), 3)
    return round(float(alphas[0]), 3) if acc[0] >= target else None


def ring_contamination():
    """Share of a target's own signal in the 3-7 px ring mean, relative to the centre pixel, for a square footprint of
    side s px convolved with a Gaussian PSF (sigma 1 px)."""
    from scipy.ndimage import gaussian_filter
    up, n = 9, 31
    out = {}
    for side in (1, 3, 5):
        img = np.zeros((n * up, n * up))
        c0 = n * up // 2
        h = side * up // 2
        img[c0 - h:c0 + h + 1, c0 - h:c0 + h + 1] = 1.0
        img = gaussian_filter(img, up * 1.0).reshape(n, up, n, up).mean(axis=(1, 3))
        cy = n // 2
        yy, xx = np.mgrid[:n, :n]
        cheb = np.maximum(abs(yy - cy), abs(xx - cy))
        out[f"{side}px"] = round(float(img[(cheb >= 3) & (cheb <= 7)].mean() / img[cy, cy]), 4)
    return out


def main():
    recs = json.load(open("results/h5_raw.json"))
    G = groups(recs)
    summ = {"ring_contamination_psf1": ring_contamination(), "groups": {}}
    for key, d in G.items():
        alphas = sorted(d["pred"], key=float)
        af = np.array([float(a) for a in alphas])
        arms = [m for m in ARMS if m in d["pred"][alphas[0]]]
        cls = d["cls"]
        acc = {m: np.array([(d["pred"][a][m] == cls).mean() for a in alphas]) for m in arms}
        g = {"n_samples": int(len(cls)), "n_spectra": int(len(np.unique(d["own"]))), "n_scenes": d["n_scenes"],
             "alphas": alphas, "acc": {m: [round(float(v), 3) for v in acc[m]] for m in arms},
             "table": {m: {a: [round(float((d["pred"][a][m] == cls).mean()), 3), boot(d["pred"][a][m] == cls, d["own"])]
                           for a in TABLE_A if a in d["pred"]} for m in arms},
             "a_half": {m: a_half(af, acc[m]) for m in arms}, "theory": {}, "strata": {}}
        for lvl, t in d["theory"].items():
            ast = t["a_star"]
            pred_curve = np.array([(af_ >= ast).mean() for af_ in af])
            agree = np.mean([((float(a) >= ast) == (d["pred"][a][f"ls_{lvl}"] == cls)).mean() for a in alphas])
            g["theory"][lvl] = {"pred_acc": [round(float(v), 3) for v in pred_curve], "per_sample_agreement": round(float(agree), 3),
                                "median_d": round(float(np.median(t["d"])), 4), "share_d_le_0": round(float((t["d"] <= 0).mean()), 3),
                                "median_delta_perp": round(float(np.median(t["delta_perp"])), 4),
                                "median_n_perp": round(float(np.median(t["n_perp"])), 4),
                                "median_a_star": round(float(np.median(ast)), 3), "a_half_pred": a_half(af, pred_curve)}
        rb = d["ring_bias"]
        lo, hi = rb <= np.percentile(rb, 50), rb >= np.percentile(rb, 75)
        for m in ("ls_none", "ls_mean", "ls_ring", "ace", "lat_none", "lat_ring", "bank_lat", "namer_ring", "namer_self"):
            if m in arms:
                g["strata"][m] = {a: {"homog": round(float((d["pred"][a][m] == cls)[lo].mean()), 3),
                                      "heterog": round(float((d["pred"][a][m] == cls)[hi].mean()), 3)} for a in ("0.1", "0.2", "0.5", "1.0")}
        g["acc_cov"] = {}
        for a in ("0.1", "0.2", "0.5"):
            if a not in d["conf"]:
                continue
            for m in arms:
                o = np.argsort(-d["conf"][a][m], kind="stable")
                hit = (d["pred"][a][m] == cls)[o]
                curve = np.cumsum(hit) / np.arange(1, len(hit) + 1)
                at = lambda q: round(float(curve[max(int(q * len(hit)) - 1, 0)]), 3)
                g["acc_cov"].setdefault(a, {})[m] = {"cov10": at(0.1), "cov25": at(0.25), "cov50": at(0.5),
                                                     "cov100": at(1.0), "area": round(float(curve.mean()), 3)}
        g["ring_bias_quartiles"] = [round(float(np.percentile(rb, p)), 3) for p in (25, 50, 75)]
        summ["groups"][key] = g
        print(key, g["n_samples"], {m: g["acc"][m][-1] for m in arms})
        print("  a_half", g["a_half"])
        print("  theory", {k: (v["a_half_pred"], v["per_sample_agreement"], v["median_d"], v["median_delta_perp"], v["median_n_perp"]) for k, v in g["theory"].items()})
    json.dump(summ, open("results/h5_summary.json", "w"), indent=1)
    plot(summ)


def plot(summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = {"ls_none": ("C0", ":"), "ls_mean": ("C0", "--"), "ls_ring": ("C0", "-"), "ace": ("C2", "--"),
            "bank_refl": ("C2", "-."), "lat_none": ("C1", ":"), "lat_mean": ("C1", "--"), "lat_ring": ("C1", "-"),
            "bank_lat": ("C1", "-."), "namer_ring": ("C3", "-"), "namer_self": ("C3", ":")}
    G = summ["groups"]
    fig, ax = plt.subplots(1, len(G), figsize=(5 * len(G), 4.2), sharey=True)
    for i, (key, g) in enumerate(G.items()):
        af = [float(a) for a in g["alphas"]]
        for m, (col, ls) in show.items():
            if m in g["acc"]:
                ax[i].plot(af, g["acc"][m], color=col, ls=ls, label=m, marker="." if m.startswith("namer") else None)
        for lvl, ls in (("ring", "-"), ("mean", "--")):
            ax[i].plot(af, g["theory"][lvl]["pred_acc"], color="grey", ls=ls, lw=0.8, label=f"theory ls_{lvl}" if i == 0 else None)
        ax[i].set_xscale("log")
        ax[i].set(title=f"{key} (n={g['n_samples']}, {g['n_spectra']} spectra)", xlabel="fill a")
    ax[0].set_ylabel("8-way class accuracy (own spectrum left out)")
    ax[-1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig("figures/h5_accuracy_vs_fill.png", dpi=130)
    recs = json.load(open("results/h5_raw.json"))
    D = groups(recs)
    fig, ax = plt.subplots(1, len(D), figsize=(5 * len(D), 4.2), sharey=True)
    for i, (key, d) in enumerate(D.items()):
        a = "0.2"
        for m, (col, ls) in show.items():
            if m in d["conf"].get(a, {}):
                o = np.argsort(-d["conf"][a][m], kind="stable")
                hit = (d["pred"][a][m] == d["cls"])[o]
                cov = np.arange(1, len(hit) + 1) / len(hit)
                ax[i].plot(cov, np.cumsum(hit) / np.arange(1, len(hit) + 1), color=col, ls=ls, label=m)
        ax[i].set(title=f"{key}, fill 0.2", xlabel="coverage (most confident first)")
    ax[0].set_ylabel("accuracy of the answered share")
    ax[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("figures/h5_accuracy_coverage_fill0.2.png", dpi=130)


if __name__ == "__main__":
    main()
