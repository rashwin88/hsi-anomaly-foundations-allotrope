"""
Light-weight per-scene evaluation of score maps without loading the scene cube (no tiers): valid / object maps from
hypersl-anomaly-bench step 0 (`stage1_datasets/hypersl_anomaly_bench/<set>_masks.npz`, exported from PrismaBench),
V2 Heavy's cached RM map and its thick-cloud class map. For methods finished on only some scenes (AutoAD), while
another GPU job holds the memory a full step 3 needs.

    C:/av/Scripts/python.exe -X utf8 partial_eval.py --method AutoAD --scenes set1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))
from app.evaluation.config import CLASS_IDS  # noqa: E402
from app.evaluation.metrics import object_hits, scene_metrics  # noqa: E402
from app.evaluation.significance import object_pauc_parts, paired_bootstrap  # noqa: E402
from step3_evaluate import FTH3_ID, FTH3_RUN, METRICS, OUT, ROOT, cached_rm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="AutoAD")
    ap.add_argument("--scenes", default="set1")
    a = ap.parse_args()
    res, parts, hits = {}, {}, {}
    for name in a.scenes.split(","):
        mk = np.load(ROOT / "hypersl_anomaly_bench" / f"{name}_masks.npz")
        valid, objects = mk["valid"].astype(bool), mk["objects"]
        cloud = np.load(FTH3_RUN / "cache/classes/prisma_bench" / f"{name}__{FTH3_ID}.npz")["cls"] == CLASS_IDS["cloud"]
        ref = cached_rm(SimpleNamespace(name=name), FTH3_ID)
        m = np.load(OUT / name / f"{a.method}.npz")["score"].astype(np.float32)
        for label, s in (("V2 Heavy", ref), (a.method, m)):
            for tag, x in (("all", s), ("excl", np.where(cloud, 0.0, s).astype(np.float32))):
                rows, meas = scene_metrics(x, valid, objects, {}, METRICS, [])
                res.setdefault(f"{label} | {tag}", {})[name] = {
                    "pauc_0.01": meas["pauc_0.01"], "roc_auc": meas["roc_auc"],
                    "caught_1_n": int(sum(r["hit_1"] for r in rows)), "caught_0.1_n": int(sum(r["hit_0.1"] for r in rows))}
                parts.setdefault(f"{label} | {tag}", {})[name] = object_pauc_parts(x, valid, objects)
                h = object_hits(x, valid, objects, 1.0)
                hits.setdefault(f"{label} | {tag}", {})[name] = {i: int(i in h) for i in parts[f"{label} | {tag}"][name]}
    boot = {}
    for tag in ("all", "excl"):
        b = paired_bootstrap(parts[f"V2 Heavy | {tag}"], parts[f"{a.method} | {tag}"], hits[f"V2 Heavy | {tag}"],
                             hits[f"{a.method} | {tag}"], n=2000, seed=0)
        boot[tag] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}
    out = {"scenes": a.scenes.split(","), "per_scene": res, "bootstrap_method_minus_v2heavy": boot}
    (HERE / f"partial_{a.method}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
