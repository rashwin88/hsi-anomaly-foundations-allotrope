"""Publication-grade summary figure for Indradhanu (hyperspectral_segformer_mae).

Parses real epoch logs from data/training_logs/indradhanu_v0.2.0.log
(append more epochs to the log and re-run — the script picks them up).

Panels:
  A — Train & validation loss (combined L1 + λ·SAM), per epoch
  B — λ ramp + LR cosine schedule (twin axis)
  C — Train L1 + SAM-degrees decoupled
  D — Per-scene benchmark table
  E — Per-scene AUC bar comparison
  F — Mean L1 vs Mean SAM scatter (per scene)
  G — Model card sidebar

Output: final design/diagrams/indradhanu_training_summary.{png,pdf}
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "data/training_logs/indradhanu_v0.2.0.log"
OUT = ROOT / "final design/diagrams/indradhanu_training_summary"

EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*train_loss:\s*([\d.eE+-]+)\s*\|\s*val_loss:\s*\[[^:]+:\s*([\d.eE+-]+)\][^|]*\|\s*avg_val:\s*([\d.eE+-]+)\s*\|\s*lr:\s*([\d.eE+-]+)"
)
DECOMP_RE = re.compile(
    r"L1:\s*([\d.eE+-]+)\s*\|\s*SAM:\s*([\d.eE+-]+)\s*rad\s*\(([\d.eE+-]+)\s*deg\)\s*\|\s*lambda:\s*([\d.eE+-]+)"
)

# Per-scene benchmark table (from user-supplied screenshot).
SCENES = [
    (1, "1186 × 1196",   992_708,    638, 0.00591, 6.64, 0.900, 0.870, 0.885),
    (2, "1216 × 1280", 1_050_895,    992, 0.01200, 2.32, 0.910, 0.771, 0.887),
    (3, "1202 × 1280",   995_483,    778, 0.01308, 2.68, 0.789, 0.855, 0.832),
    (4, "1210 × 1219", 1_005_979,  1_165, 0.01557, 3.86, 0.719, 0.750, 0.730),
]


def parse_log(path: Path):
    epochs, train, val, lr = [], [], [], []
    L1s, sam_deg, lam = [], [], []
    pending_decomp_for = None
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if m:
            ep = int(m.group(1))
            epochs.append(ep)
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
            lr.append(float(m.group(5)))
            pending_decomp_for = ep
            continue
        m2 = DECOMP_RE.search(line)
        if m2 and pending_decomp_for is not None:
            L1s.append((pending_decomp_for, float(m2.group(1))))
            sam_deg.append((pending_decomp_for, float(m2.group(3))))
            lam.append((pending_decomp_for, float(m2.group(4))))
            pending_decomp_for = None
    return {
        "epochs": np.array(epochs),
        "train": np.array(train),
        "val": np.array(val),
        "lr": np.array(lr),
        "L1": dict(L1s),
        "sam_deg": dict(sam_deg),
        "lambda": dict(lam),
    }


def main() -> None:
    d = parse_log(LOG)
    if d["epochs"].size == 0:
        raise SystemExit(f"No epoch lines parsed from {LOG}")

    ep = d["epochs"]
    train, val, lr = d["train"], d["val"], d["lr"]
    L1 = np.array([d["L1"].get(int(e), np.nan) for e in ep])
    sam = np.array([d["sam_deg"].get(int(e), np.nan) for e in ep])
    lam = np.array([d["lambda"].get(int(e), np.nan) for e in ep])

    best_idx = int(np.argmin(val))
    best_ep, best_val = int(ep[best_idx]), float(val[best_idx])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    })

    fig = plt.figure(figsize=(16, 11.5))
    gs = GridSpec(
        3, 3, figure=fig,
        height_ratios=[1.0, 1.0, 1.05],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.50, wspace=0.32,
        left=0.05, right=0.985, top=0.92, bottom=0.05,
    )

    fig.suptitle(
        "Indradhanu — Hyperspectral SegFormer-MAE: Training Trajectory & Per-Scene Benchmarks",
        fontsize=14.5, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.937,
        "Indradhanu · 165 → 32 spectral compression · 4-stage transformer encoder · two-pass checkerboard masking · α = 0.5 (L1 + SAM)",
        ha="center", fontsize=10, color="#555",
    )

    # =================== Panel A: train/val loss ===================
    axA = fig.add_subplot(gs[0, 0:2])
    axA.plot(ep, train, color="#1e40af", lw=2.0, label="Train loss")
    axA.plot(ep, val, color="#b45309", lw=2.0, linestyle="--",
             label="Validation loss (128 px)")
    axA.fill_between(ep, train, val, where=(val >= train),
                     color="#fee2e2", alpha=0.45, zorder=0,
                     label="val − train gap")
    axA.fill_between(ep, train, val, where=(val < train),
                     color="#dcfce7", alpha=0.55, zorder=0)

    axA.axvspan(1, 5, color="#fde68a", alpha=0.35, zorder=0)
    axA.text(1.2, axA.get_ylim()[1] if False else max(train.max(), val.max()),
             "warm-up\n(5 ep)", fontsize=8, color="#92400e",
             va="top", ha="left")

    # Best checkpoint marker
    axA.scatter([best_ep], [best_val], color="#16a34a", s=80, zorder=5,
                edgecolor="white", linewidth=1.5)
    axA.annotate(
        f"best val so far\nepoch {best_ep} · {best_val:.4f}",
        xy=(best_ep, best_val), xytext=(best_ep + 5, best_val + 0.012),
        fontsize=8.5, color="#166534",
        arrowprops=dict(arrowstyle="->", color="#16a34a", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7", ec="#166534", lw=0.8),
    )

    axA.set_title(f"A · Combined loss (L1 + λ·SAM)  ·  epochs 1–{int(ep.max())} of 200")
    axA.set_xlabel("Epoch")
    axA.set_ylabel("Loss")
    axA.grid(alpha=0.25)
    axA.legend(loc="lower right", frameon=False)
    axA.set_xlim(0, max(int(ep.max()) + 5, 80))

    # =================== Panel B: λ ramp + LR ===================
    axB = fig.add_subplot(gs[0, 2])
    axB.plot(ep, lam, color="#7c3aed", lw=2.2, label="SAM weight λ")
    axB.fill_between(ep, 0, lam, color="#ede9fe", alpha=0.6)
    axB.set_xlabel("Epoch")
    axB.set_ylabel("λ", color="#7c3aed")
    axB.tick_params(axis="y", colors="#7c3aed")
    axB.set_ylim(-0.02, 0.62)
    axB.spines["left"].set_color("#7c3aed")
    axB.set_title("B · λ ramp & learning-rate schedule")
    axB2 = axB.twinx()
    axB2.spines["top"].set_visible(False)
    axB2.plot(ep, lr, color="#0284c7", lw=2.0, linestyle="--",
              label="learning rate")
    axB2.set_ylabel("Learning rate", color="#0284c7")
    axB2.tick_params(axis="y", colors="#0284c7")
    axB2.set_ylim(0, max(lr) * 1.1)
    axB2.spines["right"].set_color("#0284c7")
    axB2.spines["right"].set_visible(True)

    # combined legend
    lines1, labels1 = axB.get_legend_handles_labels()
    lines2, labels2 = axB2.get_legend_handles_labels()
    axB.legend(lines1 + lines2, labels1 + labels2,
               loc="center right", frameon=False)

    # =================== Panel C: L1 + SAM (deg) decoupled ===================
    axC = fig.add_subplot(gs[1, 0:2])
    axC.plot(ep, L1, color="#0f766e", lw=2.0, label="Train L1")
    axC.set_xlabel("Epoch")
    axC.set_ylabel("L1 (reflectance units)", color="#0f766e")
    axC.tick_params(axis="y", colors="#0f766e")
    axC.set_title("C · Decoupled L1 reconstruction error & SAM (°)")
    axC.grid(alpha=0.25)

    axC2 = axC.twinx()
    axC2.spines["top"].set_visible(False)
    axC2.plot(ep, sam, color="#dc2626", lw=2.0, linestyle="--",
              label="Train SAM (°)")
    axC2.set_ylabel("SAM (degrees)", color="#dc2626")
    axC2.tick_params(axis="y", colors="#dc2626")
    axC2.spines["right"].set_color("#dc2626")
    axC2.spines["right"].set_visible(True)

    lines1, labels1 = axC.get_legend_handles_labels()
    lines2, labels2 = axC2.get_legend_handles_labels()
    axC.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", frameon=False)

    # annotate L1/SAM at first and last
    axC.annotate(f"L1: {L1[0]:.4f}", xy=(ep[0], L1[0]),
                 xytext=(ep[0] + 1, L1[0]), fontsize=8, color="#0f766e")
    axC.annotate(f"L1: {L1[-1]:.4f}", xy=(ep[-1], L1[-1]),
                 xytext=(ep[-1] - 14, L1[-1] - 0.002),
                 fontsize=8, color="#0f766e")
    axC2.annotate(f"{sam[0]:.2f}°", xy=(ep[0], sam[0]),
                  xytext=(ep[0] + 1, sam[0] - 0.4),
                  fontsize=8, color="#dc2626")
    axC2.annotate(f"{sam[-1]:.2f}°", xy=(ep[-1], sam[-1]),
                  xytext=(ep[-1] - 8, sam[-1] + 0.15),
                  fontsize=8, color="#dc2626")

    # =================== Panel D (sidebar): summary callouts ===================
    axD = fig.add_subplot(gs[1, 2])
    axD.axis("off")
    axD.set_xlim(0, 1); axD.set_ylim(0, 1)

    # stat cards
    def card(x, y, w, h, big, small, color):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02,rounding_size=0.04",
                             linewidth=1.2, edgecolor=color, facecolor="white")
        axD.add_patch(box)
        axD.text(x + w/2, y + h*0.62, big, ha="center", va="center",
                 fontsize=15, fontweight="bold", color=color)
        axD.text(x + w/2, y + h*0.22, small, ha="center", va="center",
                 fontsize=8.5, color="#374151")

    axD.text(0.5, 0.96, "Run snapshot", ha="center", fontsize=11,
             fontweight="bold", color="#1e40af")
    card(0.04, 0.74, 0.44, 0.18, f"{int(ep.max())}", "epochs logged", "#1e40af")
    card(0.52, 0.74, 0.44, 0.18, f"{best_val:.4f}", "best val (so far)", "#16a34a")
    card(0.04, 0.52, 0.44, 0.18, f"{train[-1]:.4f}", "latest train", "#b45309")
    card(0.52, 0.52, 0.44, 0.18, f"{sam[-1]:.2f}°", "latest SAM", "#dc2626")
    card(0.04, 0.30, 0.44, 0.18, f"{lr[-1]:.2e}", "latest LR", "#0284c7")
    card(0.52, 0.30, 0.44, 0.18, f"{lam[-1]:.2f}", "λ (SAM weight)", "#7c3aed")

    axD.text(0.5, 0.22, "Per-scene mean ROC-AUC", ha="center", fontsize=10,
             fontweight="bold", color="#1e293b")
    mean_auc_l1 = np.mean([s[6] for s in SCENES])
    mean_auc_sam = np.mean([s[7] for s in SCENES])
    mean_auc_cmb = np.mean([s[8] for s in SCENES])
    bars_y = 0.10
    bar_w = 0.92
    bar_h = 0.07
    axD.add_patch(FancyBboxPatch((0.04, bars_y), bar_w, bar_h,
                                 boxstyle="round,pad=0.005,rounding_size=0.02",
                                 facecolor="#f1f5f9", edgecolor="#cbd5e1", lw=0.6))
    for i, (lab, val_, color) in enumerate([
        ("L1",        mean_auc_l1,  "#1e40af"),
        ("SAM",       mean_auc_sam, "#b45309"),
        ("Combined",  mean_auc_cmb, "#166534"),
    ]):
        x0 = 0.04 + i * (bar_w / 3) + 0.01
        seg_w = bar_w / 3 - 0.02
        axD.add_patch(FancyBboxPatch(
            (x0, bars_y + 0.005), seg_w * val_, bar_h - 0.01,
            boxstyle="round,pad=0.001,rounding_size=0.005",
            facecolor=color, edgecolor="none", alpha=0.85,
        ))
        axD.text(x0 + seg_w/2, bars_y + bar_h + 0.015,
                 f"{lab}: {val_:.3f}", ha="center", fontsize=8.5, color=color,
                 fontweight="bold")

    # =================== Panel E: scene benchmark table ===================
    axE = fig.add_subplot(gs[2, 0:2])
    axE.axis("off")
    axE.set_title("E · Per-scene anomaly-detection benchmarks", loc="left", x=0.0)

    cols = ["Scene", "Spatial size", "Valid pixels", "GT anomalies",
            "Mean L1", "Mean SAM (°)", "AUC L1", "AUC SAM", "AUC Combined"]
    rows = []
    for s in SCENES:
        idx, sp, vp, ga, ml1, msam, al1, asam, acomb = s
        rows.append([
            str(idx), sp, f"{vp:,}", f"{ga:,}",
            f"{ml1:.5f}", f"{msam:.2f}",
            f"{al1:.3f}", f"{asam:.3f}", f"{acomb:.3f}",
        ])

    best_cols = []
    for s in SCENES:
        aucs = s[6:9]
        best = int(np.argmax(aucs)) + 6
        best_cols.append(best)

    table = axE.table(
        cellText=rows, colLabels=cols,
        loc="upper left",
        cellLoc="center", colLoc="center",
        bbox=[0.0, 0.18, 1.0, 0.74],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.6)
        if r == 0:
            cell.set_facecolor("#1e40af")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.16)
        else:
            cell.set_height(0.14)
            if r % 2 == 0:
                cell.set_facecolor("#f1f5f9")
        if r > 0:
            target_col = best_cols[r - 1]
            if c == target_col:
                cell.set_text_props(weight="bold", color="#166534")
                cell.set_facecolor("#dcfce7")

    axE.text(
        0.0, 0.10,
        "Bold/green = per-scene best AUC across L1 / SAM / Combined scoring "
        "(Combined = α·L1 + (1−α)·SAM, α=0.5).",
        fontsize=8.5, color="#475569", transform=axE.transAxes,
    )

    # =================== Panel F: AUC bar comparison ===================
    axF = fig.add_subplot(gs[2, 2])
    scenes_idx = np.arange(len(SCENES))
    width = 0.27
    auc_l1 = [s[6] for s in SCENES]
    auc_sam = [s[7] for s in SCENES]
    auc_cmb = [s[8] for s in SCENES]
    b1 = axF.bar(scenes_idx - width, auc_l1, width, label="L1",
                 color="#1e40af", edgecolor="white")
    b2 = axF.bar(scenes_idx, auc_sam, width, label="SAM",
                 color="#b45309", edgecolor="white")
    b3 = axF.bar(scenes_idx + width, auc_cmb, width, label="Combined",
                 color="#166534", edgecolor="white")
    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            axF.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                     f"{h:.2f}", ha="center", va="bottom", fontsize=7.5,
                     color="#374151")
    axF.set_xticks(scenes_idx)
    axF.set_xticklabels([f"S{s[0]}" for s in SCENES])
    axF.set_ylim(0.55, 1.0)
    axF.set_ylabel("ROC-AUC")
    axF.set_title("F · AUC by scoring method")
    axF.grid(axis="y", alpha=0.25)
    axF.legend(loc="lower right", frameon=False, ncol=3)

    fig.savefig(OUT.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    print(f"parsed {len(ep)} epochs from {LOG.name}")
    print(f"wrote {OUT.with_suffix('.png')}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
