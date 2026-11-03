"""Charts for the v2 AIGC Stage-III deck (15 + 5 slides).

Every chart is a matplotlib vector PDF drawn at the exact size of its slot on the slide.
Numbers are typed from the submitted report (PS11_stage3_report_allotrope21labs.pdf, 2 Nov 2026);
NUMBERS.md lists table and page for each. The band-tick chart reads the real EnMAP and AVIRIS-NG
band lists used in fine-tuning; the pretraining curve reads the run's metrics.csv (copied in figures/data).
Images (story frames, quick-looks, report figures) are already in figures/ (copied from the v1 deck and the report).
Run: PYTHONPATH=C:/pdfr C:/av/Scripts/python.exe make_figures.py
"""
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).parent / "figures"
FONTS = Path(r"C:/Users/FITT/AppData/Local/TectonicProject/Tectonic/cache/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c")
BANDS = REPO / "research/experiments/indradhanu-finetune/band_lists"

INK, MUTE, FAINT = "#111111", "#6b6b6b", "#b8b8b8"
MAN, NAT, OCHRE, PAPER = "#c0392b", "#2e7d6b", "#d4a017", "#fffff8"
BLUE, LBLUE = "#3f6f9f", "#a9c1da"
MM = 1 / 25.4


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, rcParams
    for f in ["texgyrepagella-regular.otf", "texgyrepagella-bold.otf", "texgyrepagella-italic.otf"]:
        font_manager.fontManager.addfont(str(FONTS / f))
    rcParams.update({"font.family": "TeX Gyre Pagella", "font.size": 7, "axes.edgecolor": INK,
                     "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.color": MUTE, "ytick.color": MUTE,
                     "text.color": INK, "axes.labelcolor": MUTE, "figure.facecolor": PAPER,
                     "axes.facecolor": PAPER, "savefig.facecolor": PAPER, "pdf.fonttype": 42})


def tufte(ax, left=True, bottom=True):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hbars(ax, names, vals, colors, fmt, xmax, bold=(), fs=7):
    n = len(names)
    y = list(range(n))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yi, fmt.format(v), va="center", ha="left", fontsize=fs)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK, fontsize=fs)
    for lab in ax.get_yticklabels():
        if lab.get_text() in bold:
            lab.set_fontweight("bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    tufte(ax, left=False, bottom=False)


def chance(ax, n, fs=6.3):
    ax.axvline(93, color=MUTE, lw=0.5, ls=(0, (1, 2)))
    ax.text(97, n - 0.45, "chance 93", color=MUTE, fontsize=fs, ha="left", va="center")


def save(fig, name, **kw):
    fig.savefig(OUT / name, **kw)
    import matplotlib.pyplot as plt
    plt.close(fig)


def charts():
    import matplotlib.pyplot as plt
    import numpy as np
    setup_mpl()

    # detection, thick cloud excluded, 909 objects (report Table 5 / Figure 4)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(146 * MM, 28 * MM), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = [OCHRE, "#cfcfcf", MUTE, FAINT, MAN]
    names = ["Indradhanu V2 Heavy", "Indradhanu V2.0 (FT04)", "No model (PCA-RX)", "MNF-RX", "HyperSL + same scorer"]
    hbars(a1, names, [0.189, 0.170, 0.139, 0.097, 0.095], cols, "{:.3f}", 0.235, bold=(names[0],))
    a1.set_title("pAUC, false alarms up to 1 %", fontsize=7, color=MUTE, loc="left")
    hbars(a2, [""] * 5, [449, 418, 377, 280, 369], cols, "{:.0f}", 560)
    chance(a2, 5)
    a2.set_title("objects of 909 in the top 1 % of scores", fontsize=7, color=MUTE, loc="left")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "chart_detection.pdf")

    # HyperSL bench, all pixels, same scorer (report Figure 8)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(84 * MM, 33 * MM), gridspec_kw={"width_ratios": [1.55, 1]})
    cols = [OCHRE, MUTE, MAN, "#e08b80", "#efc1ba"]
    names = ["Indradhanu V2 Heavy", "No reconstruction (PCA-RX)", "HyperSL residual", "HyperSL raw error (L2)", "HyperSL training loss"]
    hbars(a1, names, [0.175, 0.128, 0.083, 0.018, 0.004], cols, "{:.3f}", 0.25, bold=(names[0],), fs=6.4)
    a1.set_title("pAUC, all pixels", fontsize=6.4, color=MUTE, loc="left")
    hbars(a2, [""] * 5, [422, 361, 350, 130, 63], cols, "{:.0f}", 560, fs=6.4)
    chance(a2, 5, fs=5.8)
    a2.set_title("objects, top 1 %", fontsize=6.4, color=MUTE, loc="left")
    fig.tight_layout(pad=0.25, w_pad=0.6)
    save(fig, "chart_hypersl.pdf")

    # namer vs ACE, level 1 (report Figure 5)
    fig, ax = plt.subplots(figsize=(62 * MM, 44 * MM))
    xs = [0, 1, 2.7, 3.7, 5.4, 6.4]
    nm = [0.65, 0.86, 0.85, 0.94, 0.71, 0.90]
    lo = [0.55, 0.78, 0.78, 0.87, 0.63, 0.83]
    hi = [0.72, 0.92, 0.92, 0.98, 0.79, 0.94]
    ace = [0.48, 0.66, 0.61, 0.77, 0.59, 0.76]
    for x, n, l, h, a in zip(xs, nm, lo, hi, ace):
        ax.plot([x, x], [l, h], color=BLUE, lw=0.9, solid_capstyle="butt")
        ax.plot(x, n, "o", color=BLUE, ms=3.3)
        ax.plot(x, a, "o", mfc=PAPER, mec=MUTE, mew=0.8, ms=3.3)
        ax.text(x + 0.17, n, f"{n:.2f}", color=BLUE, fontsize=6, va="center")
        ax.text(x + 0.17, a, f"{a:.2f}", color=MUTE, fontsize=6, va="center")
    ax.text(-0.3, 0.65, "namer", color=BLUE, ha="right", va="center", fontsize=6.4)
    ax.text(-0.3, 0.48, "ACE", color=MUTE, ha="right", va="center", fontsize=6.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["10 %", "25 %"] * 3, fontsize=6)
    for x, lab in [(0.5, "EnMAP"), (3.2, "PRISMA"), (5.9, "AVIRIS-NG")]:
        ax.annotate(lab, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -12),
                    textcoords="offset points", ha="center", va="top", fontsize=6.4, color=INK)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(-1.9, 7.1)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=6)
    ax.set_ylabel("man-made recall", fontsize=6.4)
    tufte(ax)
    fig.tight_layout(pad=0.3)
    save(fig, "chart_namer_ace.pdf")

    # field targets, level 2 class correct (report Table 9 / Figure 7)
    meth = ["SAM", "ACE", "namer", "HSL\ndirect", "HSL\nbank"]
    sets = [("Gudalur 4 m, 5 sheets", [0.40, 0.94, 0.47, 0.33, 0.46], [0.45, 0.94, 0.47, 0.39, 0.57]),
            ("Gudalur 32 m, anecdotal", [0.15, 0.96, 0.18, 0.11, 0.18], [0.15, 0.96, 0.41, 0.11, 0.22]),
            ("HySU 0.7 m, 22 targets", [0.38, 0.44, 0.48, 0.51, 0.63], [0.58, 0.61, 0.47, 0.65, 0.77])]
    fig, axs = plt.subplots(1, 3, figsize=(96 * MM, 35 * MM), sharey=True)
    x = np.arange(5)
    for ax, (t, a, b) in zip(axs, sets):
        ax.bar(x - 0.19, a, 0.36, color=LBLUE)
        ax.bar(x + 0.19, b, 0.36, color=BLUE)
        for xi, va, vb in zip(x, a, b):
            ax.text(xi - 0.19, va + 0.02, f"{va:.2f}"[1:], ha="center", fontsize=4.6, color=MUTE)
            ax.text(xi + 0.19, vb + 0.02, f"{vb:.2f}"[1:], ha="center", fontsize=4.6, color=INK)
        ax.set_title(t, fontsize=6, color=MUTE, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(meth, fontsize=5.2, color=INK)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1.08)
        tufte(ax, left=(ax is axs[0]))
    axs[0].set_yticks([0, 0.5, 1.0])
    axs[0].tick_params(axis="y", labelsize=5.6)
    axs[0].set_ylabel("class correct", fontsize=6)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=LBLUE, label="shared library"), Patch(color=BLUE, label="+ the site's field spectra (HSL = HyperSL)")],
               loc="lower center", ncol=2, frameon=False, fontsize=5.8, bbox_to_anchor=(0.55, -0.03), handlelength=1.2)
    fig.tight_layout(pad=0.2, w_pad=0.4, rect=(0, 0.11, 1, 1))
    save(fig, "chart_field.pdf")

    # planted sub-pixel targets at 10 % fill, 8 choices (report Table 12)
    rows = [("Spectral namer, ring", [0.53, 0.39, 0.13], BLUE),
            ("Mixture bank, reflectance", [0.44, 0.30, 0.17], NAT),
            ("Ring + least squares", [0.36, 0.19, 0.15], NAT),
            ("ACE", [0.35, 0.17, 0.09], MUTE),
            ("HyperSL, ring", [0.30, 0.14, 0.06], MAN),
            ("HyperSL, mixture bank", [0.27, 0.17, 0.05], MAN),
            ("Namer, ring removed", [0.01, 0.07, 0.11], FAINT)]
    fig, ax = plt.subplots(figsize=(70 * MM, 38 * MM))
    for k, (lab, v, c) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.plot([0, 0.6], [y, y], color="#eeeeee", lw=0.5, zorder=0)
        for j, (val, mk) in enumerate(zip(v, ["o", "s", "^"])):
            ax.plot(val, y, mk, color=c, ms=3.2 if j == 0 else 2.6, mfc=c if j == 0 else PAPER, mew=0.8)
        ax.text(v[0] + 0.015, y, f"{v[0]:.2f}", fontsize=5.6, va="center", color=c if c != FAINT else MUTE)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=6, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 0.62)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.tick_params(axis="x", labelsize=5.6)
    ax.set_xlabel("class named correctly, 10 % fill", fontsize=6)
    ax.plot([], [], "o", color=INK, ms=3, label="PRISMA")
    ax.plot([], [], "s", color=INK, mfc=PAPER, ms=2.6, label="Jasper")
    ax.plot([], [], "^", color=INK, mfc=PAPER, ms=2.6, label="Urban (unseen sensor)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=5.4, handletextpad=0.2, borderpad=0.1, columnspacing=0.8)
    tufte(ax, left=False)
    fig.tight_layout(pad=0.25)
    save(fig, "chart_planted.pdf")

    # bands into 40 nm wavelength bins: two real sensor band lists, one shared token grid
    en = json.load(open(BANDS / "enmap_224.json"))["wavelengths_nm"]
    av = json.load(open(BANDS / "aviris_ng_372.json"))["wavelengths_nm"]
    fig, ax = plt.subplots(figsize=(148 * MM, 16 * MM))
    edges = np.arange(400, 2501, 40)
    for i, e in enumerate(edges[:-1]):
        if i % 2 == 0:
            ax.axvspan(e, e + 40, color="#f6e7a1", alpha=0.45, lw=0)
    for y, wl, lab in [(1.0, en, "EnMAP, 224 bands"), (0.0, av, "AVIRIS-NG, 372 bands")]:
        wl = np.asarray(wl)
        ax.vlines(wl, y - 0.32, y + 0.32, color=INK, lw=0.25)
        occ = len(set(((wl[(wl >= 400) & (wl < 2520)] - 400) // 40).astype(int)))
        ax.text(2515, y, f"{lab}\n{occ} occupied bins", fontsize=6, va="center", ha="left", color=INK, linespacing=1.1)
    ax.text(380, 0.5, "", fontsize=6)
    ax.set_xlim(370, 2760)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([])
    ax.set_xticks([400, 800, 1200, 1600, 2000, 2400])
    ax.set_xticklabels(["400 nm", "800", "1200", "1600", "2000", "2400"], fontsize=6)
    tufte(ax, left=False)
    ax.spines["bottom"].set_bounds(400, 2500)
    fig.tight_layout(pad=0.2)
    save(fig, "chart_bands.pdf")

    # atmospheric correction: core land MAE on held-out scenes vs vendor L2 (report, Harmonisation table)
    fig, ax = plt.subplots(figsize=(68 * MM, 31 * MM))
    labs = ["PRISMA\nvs ASI L2C", "EnMAP\nvs DLR L2A", "AVIRIS-NG\nvs JPL", "Tanager\nvs Planet SR"]
    before = [0.0244, 0.0098, None, None]
    after = [0.0099, 0.0086, 0.0051, None]
    x = np.arange(4)
    for xi, b, a in zip(x, before, after):
        if b is not None:
            ax.bar(xi - 0.19, b, 0.36, color=FAINT)
            ax.text(xi - 0.21, b + 0.0005, f"{b:.4f}", ha="center", fontsize=5, color=MUTE)
        if a is not None:
            off = 0.19 if b is not None else 0
            ax.bar(xi + off, a, 0.36, color=NAT)
            ax.text(xi + off + (0.03 if b else 0), a + 0.0005, f"{a:.4f}", ha="center", fontsize=5, color=INK)
    ax.plot([3, 3], [0.0061, 0.0144], color=NAT, lw=2.6, solid_capstyle="butt")
    ax.text(3.13, 0.0144, "0.0061–0.0144\n3 scenes", fontsize=5.4, va="top", ha="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FAINT, label="before harmonisation"), Patch(color=NAT, label="shipped")], loc="upper right", frameon=False, fontsize=5.4, handlelength=1, borderpad=0.1)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=5.8, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 0.027)
    ax.set_yticks([0, 0.01, 0.02])
    ax.tick_params(axis="y", labelsize=5.6)
    ax.set_ylabel("core land MAE (reflectance)", fontsize=6)
    tufte(ax)
    fig.tight_layout(pad=0.25)
    save(fig, "chart_atm_mae.pdf")

    # atmospheric correction run time (report, Measured run times)
    fig, ax = plt.subplots(figsize=(74 * MM, 31 * MM))
    rows = [("Stock ISOFIT, 12-core CPU", [("PRISMA", 62), ("EnMAP", 94)], FAINT),
            ("Ours, same 12-core CPU", [("satellite scene", (26, 44))], MUTE),
            ("Ours, H200 + 16-core VM", [("PRISMA", 13), ("EnMAP", 13), ("EnMAP held-out", 16),
                                          ("Tanager", 30), ("GHOSt", 35), ("AVIRIS-NG, 8 GB", 40)], NAT)]
    y = 0
    ticks, tl = [], []
    for grp, items, c in rows:
        for lab, v in items:
            if isinstance(v, tuple):
                ax.barh(y, v[1] - v[0], left=v[0], color=c, height=0.62)
                ax.text(v[1] + 1.5, y, f"{v[0]}–{v[1]} min", va="center", fontsize=5.6)
            else:
                ax.barh(y, v, color=c, height=0.62)
                ax.text(v + 1.5, y, f"{v} min", va="center", fontsize=5.6)
            ticks.append(y)
            tl.append(lab)
            y -= 1
        ax.text(0, y + len(items) + 0.45, grp, fontsize=5.8, color=MUTE, va="bottom")
        y -= 1.3
    ax.set_yticks(ticks)
    ax.set_yticklabels(tl, fontsize=5.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([])
    ax.set_xlim(0, 110)
    ax.set_ylim(y + 0.9, 1.4)
    tufte(ax, left=False, bottom=False)
    fig.tight_layout(pad=0.25)
    save(fig, "chart_atm_time.pdf")

    # Heavy pretraining curves (read from the run's metrics.csv)
    rows = list(csv.DictReader(open(OUT / "data/f002_metrics.csv")))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(62 * MM, 46 * MM), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(ep, [float(r["val_sam_deg"]) for r in rows], color=INK, lw=0.8)
    a1.axhline(4.06, color=MUTE, lw=0.5, ls=(0, (3, 2)))
    a1.text(79, 3.55, "smaller model's best 4.06", color=MUTE, ha="right", va="top", fontsize=5.8)
    a1.set_ylim(2.9, 9.4)
    a1.text(3, 8.84, "8.84", fontsize=6, va="center")
    a1.text(81, 4.6, "3.89", fontsize=6, va="center", color=OCHRE)
    a1.set_ylabel("val SAM (deg)", fontsize=6)
    a1.set_yticks([4, 6, 8])
    tufte(a1)
    a2.plot(ep, [float(r["train_l1"]) for r in rows], color=FAINT, lw=0.8)
    a2.plot(ep, [float(r["val_l1"]) for r in rows], color=INK, lw=0.8)
    a2.text(30, 0.0262, "train", color=MUTE, fontsize=6)
    a2.text(81, 0.0164, "val", fontsize=6, va="center")
    a2.set_ylabel("L1", fontsize=6)
    a2.set_yticks([0.015, 0.025])
    a2.set_xticks([1, 20, 40, 60, 80])
    a2.set_xlabel("pretraining epoch", fontsize=6)
    for a in (a1, a2):
        a.tick_params(labelsize=5.6)
    tufte(a2)
    fig.tight_layout(pad=0.3)
    save(fig, "chart_f002.pdf")


def chart_had100():
    """HAD100 official protocol (100 crops of 64x64, mAUC). Published rows: Li et al., IEEE TGRS 2023, Table V,
    as transcribed in research/experiments/dual-window-mahalanobis/README.md section 4b. Ours: Indradhanu V1 and V2 Heavy
    (v0.2.0) full spectrum, one value for every band setting. Dot = 200 bands; line = range over 50/100/200."""
    import matplotlib.pyplot as plt
    setup_mpl()
    rows = [  # name, (50, 100, 200), kind
        ("AETNet-KCSR", (0.9948, 0.9897, 0.9869), "pub"),
        ("KCSR", (0.9939, 0.9865, 0.9860), "pub"),
        ("CR", (0.9929, 0.9859, 0.9849), "pub"),
        ("AETNet (trained on HAD100)", (0.9925, 0.9875, 0.9818), "pub"),
        ("Indradhanu V1 + RM, local mean", (0.9748,) * 3, "ours"),
        ("Indradhanu V1 + RM", (0.9740,) * 3, "ours"),
        ("LRX", (0.9775, 0.9692, 0.9720), "pub"),
        ("Indradhanu V2 Heavy + RM", (0.9714, 0.9559, 0.9675), "ours"),  # had100-v2heavy/results/mauc_summary.json
        ("GRX", (0.9799, 0.9714, 0.9649), "pub"),
        ("AutoAD", (0.9688, 0.9661, 0.9642), "pub"),
        ("LREN", (0.9583, 0.9286, 0.9309), "pub"),
        ("RGAE", (0.9003, 0.8876, 0.8823), "pub"),
        ("Indradhanu V1, L1 error", (0.8445,) * 3, "l1"),
    ]
    fig, ax = plt.subplots(figsize=(80 * MM, 47 * MM))
    n = len(rows)
    for k, (lab, v, kind) in enumerate(rows):
        y = n - 1 - k
        c = OCHRE if kind == "ours" else (FAINT if kind == "l1" else MUTE)
        ax.plot([min(v), max(v)], [y, y], color=c, lw=0.8, solid_capstyle="round")
        ax.plot(v[2], y, "o", color=c, ms=3.4 if kind == "ours" else 2.8)
        ax.text(max(v) + 0.003, y, f"{v[2]:.4f}", va="center", fontsize=5.6, color=INK if kind == "ours" else MUTE)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=5.8, color=INK)
    for lab, r in zip(ax.get_yticklabels(), rows[::-1]):
        if r[2] == "ours":
            lab.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.83, 1.03)
    ax.set_xticks([0.85, 0.90, 0.95, 1.0])
    ax.tick_params(axis="x", labelsize=5.6)
    ax.set_xlabel("mAUC: dot at 200 bands, line over 50-200", fontsize=5.8)
    tufte(ax, left=False)
    fig.tight_layout(pad=0.25)
    save(fig, "chart_had100.pdf")


if __name__ == "__main__":
    charts()
    chart_had100()
    print("figures done")
