"""Slide-ready model roster — 7 cards in a 4×2 grid (8th slot = summary key).

Output: final design/diagrams/model_roster.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Circle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "final design/diagrams/model_roster"

# (codename, meaning, modality, bands, params, arch, norm, loss, masking,
#  use_case, color)
MODELS = [
    ("Pratibimba", "reflection",
     "Thermal", 1, "1.32M",
     "CNN AE", "Yes", "L2", "None",
     "Cheap baseline · single-pixel anomalies",
     "#0ea5e9"),
    ("Antardhana", "concealment",
     "Thermal", 1, "0.27M",
     "CNN AE", "Yes", "L2", "Random pixel",
     "Learns from spatial context",
     "#06b6d4"),
    ("Tirohita", "vanished",
     "Thermal", 1, "0.33M",
     "CNN AE", "Yes", "L1", "Random pixel",
     "Robust to training-set outliers",
     "#14b8a6"),
    ("Asanskrita", "unrefined / raw",
     "Thermal", 1, "0.27M",
     "CNN AE · 3-ch", "No (raw C)", "L1", "Channel + random",
     "Interpretable loss in Kelvin",
     "#84cc16"),
    ("Drashta", "the seer",
     "Thermal", 1, "0.27M",
     "CNN AE · 3-ch", "Yes", "L1", "Channel + random",
     "Z-score with explicit context",
     "#f59e0b"),
    ("Chakshu", "sight / eye",
     "Thermal", 1, "0.41M",
     "SegFormer MAE", "Yes", "L1", "Token MAE 50%",
     "Long-range context · diffuse hotspots",
     "#ef4444"),
    ("Indradhanu", "rainbow",
     "Hyperspectral", 165, "5.51M",
     "SegFormer MAE + spectral compressor (24-d)",
     "Per-band", "L1 + λ·SAM", "Token MAE 50%",
     "Material-specific spectral anomalies",
     "#7c3aed"),
]


def draw_card(ax, m):
    (codename, meaning, modality, bands, params,
     arch, norm, loss, mask, use_case, color) = m
    is_hyper = modality == "Hyperspectral"

    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    # outer rounded card
    bg = "#faf5ff" if is_hyper else "#ffffff"
    edge = color if is_hyper else "#e2e8f0"
    lw = 2.4 if is_hyper else 1.0
    ax.add_patch(FancyBboxPatch(
        (1, 1), 98, 98,
        boxstyle="round,pad=0.02,rounding_size=2.0",
        linewidth=lw, edgecolor=edge, facecolor=bg, zorder=1,
    ))
    # left accent bar
    ax.add_patch(FancyBboxPatch(
        (1, 1), 2.4, 98,
        boxstyle="square,pad=0", linewidth=0,
        facecolor=color, zorder=2,
    ))

    # avatar
    cx, cy, r = 14, 85, 7.2
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="white",
                        linewidth=2.5, zorder=4))
    ax.add_patch(Circle((cx, cy), r + 1.1, facecolor="none",
                        edgecolor=color, linewidth=1.2, alpha=0.35, zorder=3))
    ax.text(cx, cy + 0.3, codename[:2], ha="center", va="center",
            fontsize=22, fontweight="bold", color="white", zorder=5)

    # name + meaning
    ax.text(26, 89, codename, fontsize=22, fontweight="bold",
            color="#0f172a", va="center")
    ax.text(26, 79, f"“{meaning}”", fontsize=12, style="italic",
            color="#64748b", va="center")

    # modality chip + params chip (top-right)
    chip_w, chip_h = 22, 7
    chip_y = 86
    chip_x = 99 - 4 - chip_w
    mod_fc = "#ede9fe" if is_hyper else "#dbeafe"
    mod_ec = "#7c3aed" if is_hyper else "#1e40af"
    mod_tc = "#5b21b6" if is_hyper else "#1e3a8a"
    ax.add_patch(FancyBboxPatch(
        (chip_x, chip_y - chip_h / 2), chip_w, chip_h,
        boxstyle="round,pad=0.01,rounding_size=1.2",
        linewidth=0.8, edgecolor=mod_ec, facecolor=mod_fc, zorder=4,
    ))
    ax.text(chip_x + chip_w / 2, chip_y,
            f"{modality} · {bands} band" + ("s" if bands != 1 else ""),
            ha="center", va="center", fontsize=10, color=mod_tc,
            fontweight="bold")

    # params chip
    p_w, p_h = 16, 7
    p_x = 99 - 4 - p_w
    p_y = 75
    ax.add_patch(FancyBboxPatch(
        (p_x, p_y - p_h / 2), p_w, p_h,
        boxstyle="round,pad=0.01,rounding_size=1.2",
        linewidth=0.8, edgecolor="#cbd5e1", facecolor="#f1f5f9", zorder=4,
    ))
    ax.text(p_x + p_w / 2, p_y, f"{params} params",
            ha="center", va="center", fontsize=10, color="#334155",
            fontweight="bold")

    # horizontal rule
    ax.plot([6, 94], [68, 68], color="#e2e8f0", lw=0.8)

    # feature grid (2 cols × 2 rows)
    rows = [("ARCHITECTURE", arch), ("NORM", norm),
            ("LOSS", loss), ("MASKING", mask)]
    col_x = [6, 54]
    row_y = [58, 38]
    for i, (lab, val) in enumerate(rows):
        x = col_x[i % 2]
        y = row_y[i // 2]
        ax.text(x, y, lab, fontsize=9, color="#94a3b8", fontweight="bold")
        text_color = color if is_hyper else "#0f172a"
        weight = "bold" if is_hyper else "normal"
        # wrap longer architecture string in hyper card
        if i == 0 and is_hyper:
            ax.text(x, y - 6.5, "SegFormer MAE +", fontsize=11.5,
                    color=text_color, fontweight=weight, va="center")
            ax.text(x, y - 12.5, "spectral compressor (24-d)",
                    fontsize=11.5, color=text_color, fontweight=weight,
                    va="center")
        else:
            ax.text(x, y - 7, val, fontsize=12, color=text_color,
                    fontweight=weight, va="center")

    # use-case footer
    ax.add_patch(FancyBboxPatch(
        (6, 6), 88, 12,
        boxstyle="round,pad=0.02,rounding_size=1.2",
        linewidth=0, facecolor=color if is_hyper else "#f8fafc", alpha=0.18,
        zorder=2,
    ))
    ax.text(50, 12, use_case, ha="center", va="center",
            fontsize=11.5, color=color if is_hyper else "#0f172a",
            fontweight="bold")


def draw_legend(ax):
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.add_patch(FancyBboxPatch(
        (1, 1), 98, 98,
        boxstyle="round,pad=0.02,rounding_size=2.0",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#f8fafc",
    ))
    ax.text(50, 90, "Roster at a glance", ha="center", fontsize=17,
            fontweight="bold", color="#0f172a")
    ax.text(50, 81, "Six thermal · one hyperspectral",
            ha="center", fontsize=11, color="#475569")

    rows = [
        ("Total params", "8.4M across 7 models"),
        ("Thermal family", "6 × 1-band  ·  L1/L2 reconstruction"),
        ("Hyperspectral", "1 × 165-band  ·  L1 + λ·SAM"),
        ("Masking", "MAE-style on 5 of 7"),
        ("Bottleneck", "Indradhanu 165 → 24 channels"),
        ("Loss", "Combined α=0.5 (L1 + SAM)"),
    ]
    y = 70
    for k, v in rows:
        ax.text(8, y, k, fontsize=10.5, fontweight="bold", color="#334155")
        ax.text(42, y, v, fontsize=10.5, color="#0f172a")
        y -= 9

    # color legend
    ax.text(8, 14, "Card accent = model family", fontsize=9.5,
            color="#64748b", fontweight="bold")
    swatches = [("Plain AE", "#0ea5e9"), ("Masked AE (L2)", "#06b6d4"),
                ("Masked AE (L1)", "#14b8a6"), ("Raw-C", "#84cc16"),
                ("Norm + 3-ch", "#f59e0b"), ("SegFormer MAE", "#ef4444"),
                ("HSI MAE", "#7c3aed")]
    x = 8
    for label, c in swatches:
        ax.add_patch(Circle((x, 7), 1.5, facecolor=c, edgecolor="white",
                            linewidth=0.8))
        ax.text(x + 2.5, 7, label, fontsize=8.5, color="#334155",
                va="center")
        x += 13


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "figure.dpi": 150,
    })

    # 2 rows × 4 columns of cards (8 slots; 7 models + 1 legend)
    fig = plt.figure(figsize=(20, 11))
    outer = GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.45, 1.0, 1.0],
        hspace=0.18, left=0.02, right=0.98, top=0.97, bottom=0.025,
    )

    # title
    ax_title = fig.add_subplot(outer[0])
    ax_title.axis("off")
    ax_title.set_xlim(0, 1); ax_title.set_ylim(0, 1)
    ax_title.text(0.5, 0.72, "Allotrope · Foundation Model Roster",
                  ha="center", fontsize=26, fontweight="bold",
                  color="#0f172a")
    ax_title.text(0.5, 0.25,
                  "Seven reconstruction-based anomaly detectors  ·  "
                  "six thermal (1-band)  ·  one hyperspectral (165-band)",
                  ha="center", fontsize=13, color="#475569")

    # 2 card rows
    grid_top = GridSpec(
        1, 4, figure=fig,
        left=0.02, right=0.98,
        top=0.71, bottom=0.39, wspace=0.06,
    )
    grid_bot = GridSpec(
        1, 4, figure=fig,
        left=0.02, right=0.98,
        top=0.36, bottom=0.025, wspace=0.06,
    )

    # First row: first 4 models
    for i in range(4):
        ax = fig.add_subplot(grid_top[0, i])
        draw_card(ax, MODELS[i])

    # Second row: remaining 3 models + legend
    for i in range(3):
        ax = fig.add_subplot(grid_bot[0, i])
        draw_card(ax, MODELS[4 + i])
    ax_legend = fig.add_subplot(grid_bot[0, 3])
    draw_legend(ax_legend)

    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight",
                facecolor="white")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
