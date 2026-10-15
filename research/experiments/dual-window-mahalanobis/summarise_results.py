"""
Condense the step outputs into results/auc_summary.json, the numbers the README quotes.

    python summarise_results.py

Reads the JSONs the step scripts wrote outside the repo (PRISMA under
stage1_datasets/dual_window, public sets under public_had_datasets/dual_window_public)
and keeps only the per-scene ROC-AUC of the detectors the README compares, so a reviewer
can check every table without the data.

Gotchas:

  - "residual global" is Indradhanu residual + global Mahalanobis (RX on the residual's
    whitened top-30 PCs); "cube global" is PCA30-RX on the raw cube.
  - HAD100 appears twice: the 94-tile run (step 3, full tiles, valid pixels) and the
    official 100-crop 64×64 protocol (step 6, mAUC). Their numbers are not interchangeable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRISMA = Path(r"C:\Users\FITT\Documents\stage1_datasets\dual_window")
PUBLIC = Path(r"C:\Users\FITT\Documents\public_had_datasets\dual_window_public")

PRISMA_KEYS = ("Indradhanu combined", "Indradhanu L1", "Indradhanu SAM", "MNF-RX (product)",
               "cube global", "residual global", "residual local-mean 21/61")
HAD100_KEYS = ("Indradhanu L1", "MNF-RX (product)", "cube global", "residual global",
               "residual local-mean 5/15")


def per_scene(files: list[Path], keys: tuple[str, ...]) -> dict:
    """{scene: {detector: auc}} merged across files, for the detectors in keys."""
    merged: dict = {}
    for f in files:
        for scene, d in json.loads(f.read_text()).items():
            for k in keys:
                if isinstance(d.get(k), dict) and "auc" in d[k]:
                    merged.setdefault(scene, {})[k] = round(d[k]["auc"], 4)
    return merged


def means(scenes: dict) -> dict:
    keys = {k for d in scenes.values() for k in d}
    return {k: round(float(np.mean([d[k] for d in scenes.values() if k in d])), 4) for k in sorted(keys)}


def main() -> None:
    prisma = per_scene([PRISMA / "results.json", PRISMA / "step4.json", PRISMA / "step4_product_scores.json"],
                       PRISMA_KEYS)
    had = per_scene([PUBLIC / "had100.json", PUBLIC / "had100_mnf.json", PUBLIC / "had100_localmean.json"],
                    HAD100_KEYS)
    proto = json.loads((PUBLIC / "had100_protocol.json").read_text())
    proto_keys = [k for k in next(iter(proto.values())) if k != "gt_px"]
    out = {
        "prisma": {"per_scene": prisma, "mean": means(prisma)},
        "had100_94_tiles": {"n": len(had), "mean": means(had)},
        "had100_official_protocol": {
            "n": len(proto),
            "mAUC": {k: round(float(np.mean([v[k] for v in proto.values()])), 4) for k in proto_keys},
        },
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "auc_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["prisma"]["mean"], indent=1))


if __name__ == "__main__":
    main()
