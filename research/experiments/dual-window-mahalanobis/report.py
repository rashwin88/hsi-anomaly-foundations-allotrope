"""
Summarise a dual-window sweep's results.json into markdown tables and AUC heatmaps.

Usage: python report.py <results.json> [prior_inner prior_outer [budget ...]]

Prints, per scene: the baselines, the prior-pick window, the leave-one-scene-out
pick (best mean AUC over the other scenes) and the hindsight-best window. Saves
heatmap_<src>.png (mean AUC over scenes for every inner × outer) beside the json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def windows(res: dict, src: str) -> list[tuple[int, int]]:
    return sorted({tuple(map(int, k.split()[-1].split("/"))) for k in res if k.startswith(f"{src} dw ")})


def drop_degenerate(R: dict) -> None:
    """Remove windows whose ring was too small for most pixels, so they silently scored as global."""
    scenes = list(R)
    for k in [k for k in R[scenes[0]] if " dw " in k]:
        if np.mean([R[s][k].get("fallback", 0) for s in scenes]) > 0.5:
            print(f"(dropped {k}: ring below the minimum size, scored as global)")
            for s in scenes:
                del R[s][k]


BUDGETS = (25, 50, 100, 200)   # default false-alarm spots allowed per scene (PRISMA-sized)


def found_at(entry: dict, budget: int) -> int:
    """Most objects found at any curve cut whose false-alarm spots stay within budget."""
    return max((f for _, f, sp in entry["curve"] if sp <= budget), default=0)


def budget_tables(R: dict, scenes: list, srcs: list, prior: tuple, budgets: tuple) -> None:
    """Objects found at equal false-alarm budgets: baselines, prior pick, leave-one-scene-out pick."""
    mid = budgets[len(budgets) // 2]
    total = sum(next(iter(R[s].values()))["objects_total"] for s in scenes)
    print(f"\n### Objects found (of {total}) with at most N false-alarm spots per scene\n")
    print("| detector | " + " | ".join(f"N={b}" for b in budgets) + " |")
    print("|---" * (len(budgets) + 1) + "|")
    rows = {k: [sum(found_at(R[s][k], b) for s in scenes) for b in budgets]
            for k in R[scenes[0]] if " dw " not in k}
    for src in srcs:
        ws = [k for k in R[scenes[0]] if k.startswith(f"{src} dw ")]
        rows[f"{src} dw {prior[0]}/{prior[1]} (prior pick)"] = [
            sum(found_at(R[s][f"{src} dw {prior[0]}/{prior[1]}"], b) for s in scenes) for b in budgets]
        loso, picks = [], []
        for b in budgets:
            tot = 0
            for s in scenes:
                others = [o for o in scenes if o != s]
                w = max(ws, key=lambda k: sum(found_at(R[o][k], b) for o in others))
                tot += found_at(R[s][w], b)
                picks.append(w.split()[-1])
            loso.append(tot)
        rows[f"{src} dw leave-one-scene-out (picked: {', '.join(sorted(set(picks)))})"] = loso
        best = max(ws, key=lambda k: sum(found_at(R[s][k], mid) for s in scenes))
        rows[f"{src} dw best overall at N={mid} ({best.split()[-1]}, hindsight)"] = [
            sum(found_at(R[s][best], b) for s in scenes) for b in budgets]
    for k, v in rows.items():
        print(f"| {k} | " + " | ".join(f"{x} ({x / total:.0%})" for x in v) + " |")

    for src in srcs:
        ws = [k for k in R[scenes[0]] if k.startswith(f"{src} dw ")]
        print(f"\n{src}: objects found at N={mid} per window: "
              + ", ".join(f"{k.split()[-1]}:{sum(found_at(R[s][k], mid) for s in scenes)}" for k in ws))


def main() -> None:
    path = Path(sys.argv[1])
    prior = tuple(map(int, sys.argv[2:4])) if len(sys.argv) > 3 else (11, 31)
    budgets = tuple(map(int, sys.argv[4:])) or BUDGETS
    R = json.loads(path.read_text())
    drop_degenerate(R)
    scenes = list(R)
    srcs = [s for s in ("residual", "cube") if any(k.startswith(f"{s} dw") for k in R[scenes[0]])]
    first = next(iter(R[scenes[0]].values()))
    cut = next(k for k in first if k.startswith("top1"))
    for metric in ("auc", "ap"):
        print(f"\n### {metric.upper()} per scene\n")
        print("| detector | " + " | ".join(scenes) + " | mean |")
        print("|---" * (len(scenes) + 2) + "|")
        rows = [(k, k) for k in R[scenes[0]] if " dw " not in k]
        for src in srcs:
            ws = windows(R[scenes[0]], src)
            rows.append((f"{src} dw {prior[0]}/{prior[1]} (prior pick)", f"{src} dw {prior[0]}/{prior[1]}"))
            loso, best = [], []
            for s in scenes:
                others = [o for o in scenes if o != s]
                w = max(ws, key=lambda w: np.mean([R[o][f"{src} dw {w[0]}/{w[1]}"][metric] for o in others]))
                loso.append((w, R[s][f"{src} dw {w[0]}/{w[1]}"][metric]))
                b = max(ws, key=lambda w: R[s][f"{src} dw {w[0]}/{w[1]}"][metric])
                best.append((b, R[s][f"{src} dw {b[0]}/{b[1]}"][metric]))
            for label, vals in ((f"{src} dw leave-one-scene-out", loso), (f"{src} dw hindsight best", best)):
                cells = [f"{v:.3f} ({w[0]}/{w[1]})" for w, v in vals]
                print(f"| {label} | " + " | ".join(cells) + f" | {np.mean([v for _, v in vals]):.3f} |")
        for label, key in rows:
            if key not in R[scenes[0]]:
                continue
            vals = [R[s][key][metric] for s in scenes]
            print(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {np.mean(vals):.3f} |")

    print(f"\n### Objects found / false-alarm spots at the {cut} cut (2 px neutral zone, all scenes)\n")
    keys = [k for k in R[scenes[0]] if " dw " not in k] + [f"{s} dw {prior[0]}/{prior[1]}" for s in srcs]
    total = sum(R[s][keys[0]]["objects_total"] for s in scenes)
    print("| detector | objects found | false-alarm spots |\n|---|---|---|")
    for k in keys:
        f = sum(R[s][k][cut]["found"] for s in scenes)
        sp = sum(R[s][k][cut]["spots"] for s in scenes)
        print(f"| {k} | {f}/{total} ({f / total:.0%}) | {sp:,} |")

    if "curve" in first:
        budget_tables(R, scenes, srcs, prior, budgets)

    for src in srcs:
        ws = windows(R[scenes[0]], src)
        inn, out = sorted({w[0] for w in ws}), sorted({w[1] for w in ws})
        M = np.full((len(inn), len(out)), np.nan)
        for i, o in ws:
            M[inn.index(i), out.index(o)] = np.mean([R[s][f"{src} dw {i}/{o}"]["auc"] for s in scenes])
        g = np.mean([R[s][f"{src} global"]["auc"] for s in scenes])
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(M, cmap="viridis", origin="lower")
        for a in range(len(inn)):
            for b in range(len(out)):
                if np.isfinite(M[a, b]):
                    ax.text(b, a, f"{M[a, b]:.3f}", ha="center", va="center", fontsize=7,
                            color="w" if M[a, b] < np.nanmean(M) else "k")
        ax.set_xticks(range(len(out)), out); ax.set_yticks(range(len(inn)), inn)
        ax.set_xlabel("outer window (px)"); ax.set_ylabel("inner / guard window (px)")
        ax.set_title(f"{src}: mean ROC-AUC over {len(scenes)} scenes (global = {g:.3f})")
        fig.colorbar(im); fig.tight_layout()
        fig.savefig(path.parent / f"heatmap_{src}.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
