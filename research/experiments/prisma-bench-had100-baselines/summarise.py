"""
results.json (step 3) -> results.md: the slide table (all pixels / thick cloud excluded), paired differences against
V2 Heavy with 95 % object-bootstrap intervals, and per-scene pAUC.

    C:/av/Scripts/python.exe -X utf8 summarise.py [--tag _set1]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = "Indradhanu V2 Heavy (FTH3)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    res = json.loads((HERE / f"results{a.tag}.json").read_text())
    S, B, O = res["summary"], res["bootstrap_vs_v2heavy"], res["own_intervals"]
    bases = [k for k in S if not k.endswith(" | excl")]
    bases.sort(key=lambda k: -S[k + " | excl"]["pauc_0.01"])
    n = S[REF]["objects"]
    L = [f"Objects: {n}. Chance (analytic): top 1 % {S[REF]['chance_1_n']:.0f}, top 0.1 % {S[REF]['chance_0.1_n']:.1f} objects.",
         "Pixel measures: mean over scenes. Interval: 95 % object bootstrap (2,000 draws, seed 0).", "",
         "| method | pAUC all | pAUC excl. [95 %] | top 1 % all | top 1 % excl. | top 0.1 % all | top 0.1 % excl. | ROC-AUC all | ROC-AUC excl. |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in bases:
        s, e = S[k], S[k + " | excl"]
        ci = O[k + " | excl"]["pauc_ci"]
        L.append(f"| {k} | {s['pauc_0.01']:.3f} | {e['pauc_0.01']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | {s['caught_1_n']} | "
                 f"{e['caught_1_n']} | {s['caught_0.1_n']} | {e['caught_0.1_n']} | {s['roc_auc']:.3f} | {e['roc_auc']:.3f} |")
    L += ["", "Paired difference, method minus V2 Heavy (same cloud setting), 95 % interval:", "",
          "| method | cloud | d pAUC [95 %] | d objects top 1 % (pts) [95 %] | d objects top 0.1 % (pts) [95 %] |",
          "|---|---|---|---|---|"]
    for k in bases:
        for suf, tag in (("", "all"), (" | excl", "excl.")):
            b = B.get(k + suf)
            if not b:
                continue
            L.append(f"| {k} | {tag} | {b['pauc_diff']:+.4f} [{b['pauc_ci'][0]:+.4f}, {b['pauc_ci'][1]:+.4f}] | "
                     f"{100 * b['caught_1_diff']:+.1f} [{100 * b['caught_1_ci'][0]:+.1f}, {100 * b['caught_1_ci'][1]:+.1f}] | "
                     f"{100 * b['caught_0.1_diff']:+.1f} [{100 * b['caught_0.1_ci'][0]:+.1f}, {100 * b['caught_0.1_ci'][1]:+.1f}] |")
    scenes = sorted(S[REF]["per_scene"])
    L += ["", "Per scene, thick cloud excluded: pAUC / objects top 1 %:", "",
          "| method | " + " | ".join(scenes) + " |", "|---|" + "---|" * len(scenes)]
    for k in bases:
        ps = S[k + " | excl"]["per_scene"]
        L.append(f"| {k} | " + " | ".join(f"{ps[s]['pauc_0.01']:.3f} / {ps[s]['caught_1_n']}" for s in scenes) + " |")
    if res.get("incomplete"):
        L += ["", "Not on every scene (left out above): " + ", ".join(res["incomplete"])]
    (HERE / f"results{a.tag}.md").write_text("\n".join(L) + "\n", encoding="utf8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
