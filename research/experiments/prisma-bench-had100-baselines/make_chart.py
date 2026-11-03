"""
Deck charts for this experiment, in the style of slide A6's HAD100 dot chart (make_figures.chart_had100: one row per
method sorted best to worst, value to the right, ours in ochre, others grey, same fonts and 80 x 47 mm slot).

  chart_prisma_had100.pdf / .png          pAUC (FPR <= 1 %), thick cloud excluded; dot = pooled value (mean over the
                                          4 scenes), line = 95 % object-bootstrap interval
  chart_prisma_had100_objects.pdf / .png  objects caught in the top 1 % (of 909), cloud excluded; line = 95 % interval;
                                          dotted chance line

Reads results.json (step3). Writes here and into research/presentation/aigc-stage3/v2/figures under new names only.
    PYTHONPATH=C:/pdfr C:/av/Scripts/python.exe -X utf8 make_chart.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DECK = HERE.parents[2] / "research/presentation/aigc-stage3/v2"
sys.path.insert(0, str(DECK))
import make_figures as mf  # noqa: E402

NAMES = {  # results.json label -> slide label, kind
    "Indradhanu V2 Heavy (FTH3)": ("Indradhanu V2 Heavy + RM", "ours"),
    "Indradhanu V2.0 (FT04)": ("Indradhanu V2.0 (FT04) + RM", "ours"),
    "Indradhanu V1 (v0.2.0)": ("Indradhanu V1 + RM", "ours"),
    "PCA-RX (no model)": ("PCA-RX (no model)", "ref"),
    "MNF-RX": ("MNF-RX (product classical)", "ref"),
    "GRX": ("GRX, all bands", "pub"),
    "GRX_50b": ("GRX, first 50 bands", "pub"),
    "GRX_100b": ("GRX, first 100 bands", "pub"),
    "LRX_13_29": ("LRX 13/29", "pub"),
    "LRX_5_15": ("LRX 5/15", "pub"),
    "KCSR_3_29": ("KCSR 3/29 (approx.)", "pub"),
    "AutoAD": ("AutoAD", "pub"),
}


def rows_for(res: dict, metric: str) -> list:
    S, own = res["summary"], res["own_intervals"]
    crs = [k for k in S if k.startswith("CR_lam") and k.endswith(" | excl")]
    best_cr = max(crs, key=lambda k: S[k]["pauc_0.01"]) if crs else None   # chosen on pAUC for both panels
    out = []
    for k, v in S.items():
        if not k.endswith(" | excl"):
            continue
        base = k[:-7]
        if base.startswith("CR_lam"):
            if k != best_cr:
                continue
            name, kind = (f"CR 5/29, \u03bb {base[6:]}", "pub")
        elif base in NAMES:
            name, kind = NAMES[base]
        else:
            continue
        if metric == "pauc":
            out.append((name, v["pauc_0.01"], own[k]["pauc_ci"], kind))
        else:
            out.append((name, v["caught_1_n"], own[k]["caught_1_n_ci"], kind))
    return sorted(out, key=lambda r: -r[1])


def dot_chart(rows, xlim, xticks, fmt, xlabel, name, chance=None):
    import matplotlib.pyplot as plt
    mf.setup_mpl()
    fig, ax = plt.subplots(figsize=(80 * mf.MM, 47 * mf.MM))
    n = len(rows)
    span = xlim[1] - xlim[0]
    for k, (lab, v, ci, kind) in enumerate(rows):
        y = n - 1 - k
        c = mf.OCHRE if kind == "ours" else mf.MUTE
        ax.plot(ci, [y, y], color=c, lw=0.8, solid_capstyle="round")
        ax.plot(v, y, "o", color=c, ms=3.4 if kind == "ours" else 2.8)
        ax.text(ci[1] + span * 0.015, y, fmt.format(v), va="center", fontsize=5.6,
                color=mf.INK if kind == "ours" else mf.MUTE)
    if chance is not None:
        ax.axvline(chance, color=mf.MUTE, lw=0.5, ls=(0, (1, 2)))
        ax.text(chance + span * 0.01, -0.9, f"chance {chance:.0f}", color=mf.MUTE, fontsize=5.4, ha="left", va="center")
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=5.8, color=mf.INK)
    for lab, r in zip(ax.get_yticklabels(), rows[::-1]):
        if r[3] == "ours":
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(-1.3 if chance is not None else -0.6, n - 0.4)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=5.6)
    ax.set_xlabel(xlabel, fontsize=5.8, loc="right")
    mf.tufte(ax, left=False)
    fig.tight_layout(pad=0.25)
    for d in (HERE, DECK / "figures"):
        fig.savefig(d / f"{name}.pdf")
        fig.savefig(d / f"{name}.png", dpi=300)
    plt.close(fig)


def main() -> None:
    res = json.loads((HERE / "results.json").read_text())
    rp = rows_for(res, "pauc")
    hi = max(r[2][1] for r in rp)
    dot_chart(rp, (0, round(hi + 0.045, 2)), [0, 0.05, 0.10, 0.15, 0.20], "{:.3f}",
              "pAUC (FPR \u2264 1 %), cloud excl.; dot pooled, line 95 % CI",
              "chart_prisma_had100")
    ro = rows_for(res, "objects")
    chance = res["summary"]["Indradhanu V2 Heavy (FTH3) | excl"]["chance_1_n"]
    hi = max(r[2][1] for r in ro)
    dot_chart(ro, (0, hi + 80), [0, 100, 200, 300, 400, 500], "{:.0f}",
              "objects of 909 in top 1 %, cloud excl.; line 95 % CI",
              "chart_prisma_had100_objects", chance=chance)
    print("charts written")


if __name__ == "__main__":
    main()
