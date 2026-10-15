"""
Head-to-head ROC-AUC: Indradhanu residual-Mahalanobis against every other scorer.

Usage: python compare_auc.py <results.json> [<more.json> ...]

The files are merged per scene, so a detector found in any of them is compared.
For each detector this prints its mean AUC, the per-scene win/loss count of
"residual global" against it, and a two-sided Wilcoxon signed-rank p-value on
the paired per-scene AUCs. With 4 PRISMA scenes the smallest possible p is
0.125, so there only the win count is meaningful.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import wilcoxon

REF = "residual global"


def main() -> None:
    R: dict = {}
    for f in sys.argv[1:]:
        for scene, d in json.loads(open(f).read()).items():
            R.setdefault(scene, {}).update(d)
    scenes = [s for s in R if REF in R[s]]
    keys = [k for k in R[scenes[0]] if isinstance(R[scenes[0]][k], dict) and "auc" in R[scenes[0]][k]
            and (" dw " not in k and " seg" not in k and "local-mean" not in k)]
    ref = np.array([R[s][REF]["auc"] for s in scenes])
    print(f"{len(scenes)} scenes; reference = Indradhanu residual + global Mahalanobis "
          f"(mean AUC {ref.mean():.3f})\n")
    print("| detector | mean AUC | reference wins / losses | Wilcoxon p |\n|---|---|---|---|")
    for k in sorted(keys, key=lambda k: -np.mean([R[s][k]["auc"] for s in scenes])):
        a = np.array([R[s][k]["auc"] for s in scenes])
        if k == REF:
            print(f"| **{k}** | **{a.mean():.3f}** | - | - |")
            continue
        p = wilcoxon(ref, a).pvalue if np.any(ref != a) else 1.0
        print(f"| {k} | {a.mean():.3f} | {(ref > a).sum()} / {(ref < a).sum()} | {p:.2g} |")


if __name__ == "__main__":
    main()
