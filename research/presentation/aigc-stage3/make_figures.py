"""Figures for the AIGC Stage-III deck.

images(): copy, crop and downscale the product screenshots and story frames into figures/.
charts(): every chart, drawn with matplotlib as a vector PDF at the exact size of its slot on the slide.
Numbers in the charts are typed from the sources listed in NUMBERS.md (F002 curves read from its metrics.csv).
Run with the pv2 venv (PIL, matplotlib): python make_figures.py
"""
import csv
import shutil
from pathlib import Path

from PIL import Image

SP = Path(r"C:/Users/FITT/AppData/Local/Temp/claude/c--Users-FITT-OneDrive-Documents-GitHub-hsi-anomaly-foundations-allotrope/788c753e-9469-41c6-b3b0-db0026d037fc/scratchpad")
REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
STORY = SP / "final/story/allotrope_story_c7648a5f_56b746f0_prisma_default"
STORY_A = SP / "final/story/allotrope_story_9021b6e5_fae2bccf_aviris_oil_plastic"
FONTS = Path(r"C:/Users/FITT/AppData/Local/TectonicProject/Tectonic/cache/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c")

INK, MUTE, FAINT = "#111111", "#6b6b6b", "#b8b8b8"
MAN, OCHRE, PAPER = "#c0392b", "#d4a017", "#fffff8"
MM = 1 / 25.4


def save(src, name, maxw=1400, crop=None):
    im = Image.open(src).convert("RGB")
    if crop:
        w, h = im.size
        im = im.crop((int(crop[0] * w), int(crop[1] * h), int(crop[2] * w), int(crop[3] * h)))
    if im.width > maxw:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    im.save(OUT / name, quality=88)


def images():
    for i in range(1, 8):
        save(STORY / f"key{i}.png", f"story_prisma_k{i}.jpg", 700, crop=(0, 0.035, 1, 0.955))
        save(STORY_A / f"key{i}.png", f"story_aviris_k{i}.jpg", 700, crop=(0, 0.03, 1, 0.96))
    save(SP / "final/seg-prisma_v2.png", "seg_prisma_panel.jpg", 1100, crop=(0.405, 0.61, 0.96, 0.98))
    save(SP / "final/seg-enmap_v2.png", "seg_enmap_panel.jpg", 1100, crop=(0.405, 0.61, 0.96, 0.98))
    save(SP / "final/flowB/prisma-1-open.png", "match_prisma.jpg", 1800)
    for q in ["prisma_manmade", "tanager_manmade"]:
        save(SP / f"final/flow/{q}_quicklook.png", f"ql_{q}.jpg", 1000)
    shutil.copy(REPO / "research/experiments/spectral-namer/figures/namer_network.pdf", OUT / "namer_network.pdf")
    (OUT / "data").mkdir(exist_ok=True)
    shutil.copy(REPO / "research/experiments/indradhanu-se-pretrain/F002-heavy/metrics.csv", OUT / "data/f002_metrics.csv")


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


def hbars(ax, names, vals, colors, fmt, xmax, bold=()):
    n = len(names)
    y = list(range(n))[::-1]
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yi, fmt.format(v), va="center", ha="left", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=INK)
    for lab in ax.get_yticklabels():
        if lab.get_text() in bold:
            lab.set_fontweight("bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    tufte(ax, left=False, bottom=False)


def chance(ax, n):
    ax.axvline(93, color=MUTE, lw=0.5, ls=(0, (1, 2)))
    ax.text(97, n - 0.45, "chance 93", color=MUTE, fontsize=6.3, ha="left", va="center")


def charts():
    import matplotlib.pyplot as plt
    setup_mpl()

    # 1. scores on the same reconstruction: mean ROC-AUC, Indradhanu v0.2.0, Set-1..4
    fig, ax = plt.subplots(figsize=(72 * MM, 46 * MM))
    hbars(ax, ["RM, ring 21/61", "RM, global mean", "PCA30-RX on the cube", "MNF-RX on the cube",
               "L1 size", "combined (old default)", "SAM size"],
          [0.854, 0.839, 0.831, 0.752, 0.702, 0.662, 0.570],
          [OCHRE, OCHRE, MUTE, MUTE, FAINT, FAINT, FAINT], "{:.3f}", 1.0, bold=("RM, ring 21/61",))
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "chart_rm.pdf")
    plt.close(fig)

    # 2. detection, cloud excluded, 909 objects
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(146 * MM, 42 * MM), gridspec_kw={"width_ratios": [1.3, 1]})
    cols = [OCHRE, FAINT, FAINT, FAINT, FAINT]
    hbars(a1, ["V2 Heavy", "V2 FT04", "PCA-RX, no model", "MNF-RX", "HyperSL + RM"],
          [0.1892, 0.170, 0.1390, 0.0966, 0.0947], cols, "{:.3f}", 0.235, bold=("V2 Heavy",))
    a1.set_title("pAUC, false-alarm rate up to 1 %", fontsize=7, color=MUTE, loc="left")
    hbars(a2, [""] * 5, [449, 418, 377, 280, 369], cols, "{:.0f}", 560)
    chance(a2, 5)
    a2.set_title("objects of 909 in the top 1 % of scores", fontsize=7, color=MUTE, loc="left")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(OUT / "chart_detection.pdf")
    plt.close(fig)

    # 3. HyperSL bench, same scorer, no exclusion
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(146 * MM, 46 * MM), gridspec_kw={"width_ratios": [1.35, 1]})
    cols = [OCHRE, MUTE, MAN, FAINT, MAN, MAN]
    hbars(a1, ["V2 Heavy residual + RM", "no model (PCA-RX)", "HyperSL residual + RM", "MNF-RX",
               "HyperSL L2 residual", "HyperSL training loss"],
          [0.1749, 0.1281, 0.0832, 0.0750, 0.0175, 0.0039], cols, "{:.3f}", 0.22, bold=("V2 Heavy residual + RM",))
    a1.set_title("pAUC, false-alarm rate up to 1 %", fontsize=7, color=MUTE, loc="left")
    hbars(a2, [""] * 6, [422, 361, 350, 249, 130, 63], cols, "{:.0f}", 540)
    chance(a2, 6)
    a2.set_title("objects of 909 in the top 1 %", fontsize=7, color=MUTE, loc="left")
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(OUT / "chart_hypersl.pdf")
    plt.close(fig)

    # 4. namer vs ACE: man-made recall at ACE's natural false-alarm rate, 95 % CI over library spectra
    fig, ax = plt.subplots(figsize=(84 * MM, 54 * MM))
    xs = [0, 1, 2.7, 3.7, 5.4, 6.4]
    nm = [0.65, 0.86, 0.85, 0.94, 0.71, 0.90]
    lo = [0.55, 0.78, 0.78, 0.87, 0.63, 0.83]
    hi = [0.72, 0.92, 0.92, 0.98, 0.79, 0.94]
    ace = [0.48, 0.66, 0.61, 0.77, 0.59, 0.76]
    for x, n, l, h, a in zip(xs, nm, lo, hi, ace):
        ax.plot([x, x], [l, h], color=MAN, lw=0.9, solid_capstyle="butt")
        ax.plot(x, n, "o", color=MAN, ms=3.6)
        ax.plot(x, a, "o", mfc=PAPER, mec=MUTE, mew=0.8, ms=3.6)
        ax.text(x + 0.16, n, f"{n:.2f}", color=MAN, fontsize=6.5, va="center")
        ax.text(x + 0.16, a, f"{a:.2f}", color=MUTE, fontsize=6.5, va="center")
    ax.text(-0.3, 0.65, "namer", color=MAN, ha="right", va="center", fontsize=7)
    ax.text(-0.3, 0.48, "ACE", color=MUTE, ha="right", va="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["0.1", "0.25"] * 3)
    for x, lab in [(0.5, "EnMAP"), (3.2, "PRISMA"), (5.9, "AVIRIS-NG")]:
        ax.annotate(lab, (x, 0), xycoords=("data", "axes fraction"), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top", fontsize=7, color=INK)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(-1.5, 7.1)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("target fill (share of the pixel), by sensor", labelpad=14)
    tufte(ax)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "chart_namer_ace.pdf")
    plt.close(fig)

    # 5. Heavy pretraining curves (read from the run's metrics.csv)
    rows = list(csv.DictReader(open(OUT / "data/f002_metrics.csv")))
    ep = [int(r["epoch"]) for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(72 * MM, 54 * MM), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(ep, [float(r["val_sam_deg"]) for r in rows], color=INK, lw=0.8)
    a1.axhline(4.06, color=MUTE, lw=0.5, ls=(0, (3, 2)))
    a1.text(79, 3.62, "F001 best 4.06 (epoch 100)", color=MUTE, ha="right", va="top", fontsize=6.5)
    a1.set_ylim(3.0, 9.4)
    a1.text(3, 8.84, "8.84", fontsize=6.5, va="center")
    a1.text(81, 3.89, "3.89", fontsize=6.5, va="center", color=OCHRE)
    a1.set_ylabel("val SAM (deg)")
    a1.set_yticks([4, 6, 8])
    tufte(a1)
    a2.plot(ep, [float(r["train_l1"]) for r in rows], color=FAINT, lw=0.8)
    a2.plot(ep, [float(r["val_l1"]) for r in rows], color=INK, lw=0.8)
    a2.text(30, 0.0262, "train", color=MUTE, fontsize=6.5)
    a2.text(81, 0.0164, "val 0.0164", fontsize=6.5, va="center")
    a2.set_ylabel("L1")
    a2.set_yticks([0.015, 0.025])
    a2.set_xticks([1, 20, 40, 60, 80])
    a2.set_xlabel("epoch")
    tufte(a2)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "chart_f002.pdf")
    plt.close(fig)

    # 6. evolution: cloud-excluded pAUC by model generation (shipped in ink, not shipped faint)
    fig, ax = plt.subplots(figsize=(146 * MM, 24 * MM))
    gens = ["V1 v0.2.0", "FT04 (V2.0)", "FT07", "FT08", "FT09b", "FT09c", "Heavy FTH3 (V2.1)"]
    v = [0.165, 0.170, 0.165, 0.164, 0.176, 0.172, 0.189]
    shipped = [True, True, False, False, False, False, True]
    ax.axhline(0.139, color=FAINT, lw=0.5)
    ax.text(-0.35, 0.139, "no model (PCA-RX) 0.139", color=MUTE, fontsize=6.3, va="bottom", ha="left")
    ax.plot([0, 1, 6], [0.165, 0.170, 0.189], color=INK, lw=0.8)
    for i, (g, val, s) in enumerate(zip(gens, v, shipped)):
        c = OCHRE if i == 6 else (INK if s else FAINT)
        ax.plot(i, val, "o", color=c, ms=4 if s else 3)
        if s:
            ax.text(i, val + 0.0035, f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
        else:
            ax.text(i, val - 0.0035, f"{val:.3f}", ha="center", va="top", fontsize=6.5, color=MUTE)
    ax.set_xticks(range(7))
    ax.set_xticklabels(gens, fontsize=6.5)
    for lab, s in zip(ax.get_xticklabels(), shipped):
        lab.set_color(INK if s else MUTE)
    ax.set_ylim(0.135, 0.20)
    ax.set_xlim(-0.4, 6.5)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0)
    tufte(ax, left=False, bottom=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "chart_generations.pdf")
    plt.close(fig)

    # 7. sparkline for the journey
    fig, ax = plt.subplots(figsize=(42 * MM, 13 * MM))
    ax.plot(range(7), v, color=INK, lw=0.7, marker="o", ms=1.6)
    ax.plot(6, 0.189, "o", color=OCHRE, ms=3.2)
    ax.axhline(0.139, color=FAINT, lw=0.5)
    ax.text(6.3, 0.189, "0.189", fontsize=6, va="center")
    ax.text(0, 0.1405, "no model 0.139", fontsize=5.6, color=MUTE, va="bottom")
    ax.text(-0.2, 0.165, "0.165", fontsize=5.6, va="center", ha="right")
    ax.set_xlim(-1.4, 7.4)
    ax.set_ylim(0.134, 0.195)
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "chart_spark.pdf", transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    images()
    charts()
    print("figures done")
