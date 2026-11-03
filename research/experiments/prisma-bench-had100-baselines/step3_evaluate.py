"""
Step 3: measure the HAD100 methods' score maps (steps 1-2) with the evaluator's own measures beside Indradhanu V2 Heavy
(FTH3 epoch 5), V2.0 (FT04), V1 (v0.2.0) and the two classical references (PCA-RX, MNF-RX), on the same 909 objects,
all pixels and with thick cloud excluded by FTH3 epoch 5's own class map (scored 0 after scoring, as in the FTH3
report). Pattern and helpers as research/experiments/hypersl-anomaly-bench/step2_evaluate.py.

    C:/av/Scripts/python.exe -X utf8 step3_evaluate.py

Reads   OUT/<scene>/<method>.npz (steps 1-2; whatever exists), the FTH3 run's cache (C:/es/eval/fth2_ftb2) for FTH3 /
        FT04 RM and the class maps, the stage-3 run's cache (C:/es/eval/stage3) for v0.2.0 RM (found by matching its
        per-scene pAUC in per_scene.csv, since that run predates the current cache key).
Writes  results.json, results.md, per_scene.csv beside this file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from app.evaluation.config import CLASS_IDS, EvaluationConfig  # noqa: E402
from app.evaluation.datasets import PRISMA_SCENES, PrismaBench  # noqa: E402
from app.evaluation.metrics import object_hits, scene_metrics  # noqa: E402
from app.evaluation.significance import object_pauc_parts, paired_bootstrap  # noqa: E402

ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT = ROOT / "had100_baselines"
FTH3_RUN = Path(r"C:\es\eval\fth2_ftb2")
STAGE3 = Path(r"C:\es\eval\stage3")
FTH3_ID, FT04_ID, V1_ID = "FTH3_ep5-fefed77b7a", "FT04_shipped-e1349850b8", "v0.2.0-1dc63318c6"
METRICS = [SimpleNamespace(kind="roc_auc"), SimpleNamespace(kind="pauc", max_fpr=0.01),
           SimpleNamespace(kind="tpr_at", fpr=0.001), SimpleNamespace(kind="objects_caught", top_pct=[1.0, 0.1])]
REF = "Indradhanu V2 Heavy (FTH3)"
HAD_METHODS = ["GRX", "GRX_50b", "GRX_100b", "LRX_13_29", "LRX_5_15", "CR_lam0.001", "CR_lam0.01", "CR_lam0.1",
               "CR_lam1", "KCSR_3_29", "AutoAD"]


def cached_rm(scene, model_id: str) -> np.ndarray:
    """A model's cached 'RM' map from the FTH3 run, located by the runner's own cache key."""
    cfg = EvaluationConfig.from_file(REPO / "configs/eval/fth3.json")
    he5 = ROOT / PRISMA_SCENES[scene.name][0]
    fp = f"{he5.stat().st_size}-{he5.stat().st_mtime_ns}"
    spec = next(s for s in cfg.scorers if s.label == "RM")
    key = hashlib.sha1((spec.model_dump_json() + "|" + fp).encode()).hexdigest()[:10]
    return np.load(FTH3_RUN / "cache/scores/prisma_bench" / scene.name / model_id / f"{key}.npz")["score"].astype(np.float32)


def v1_rm(scene) -> np.ndarray | None:
    """v0.2.0's residual-Mahalanobis map from the stage-3 cache: the file whose pAUC matches per_scene.csv."""
    want = None
    with open(STAGE3 / "per_scene.csv", newline="") as f:
        for r in csv.DictReader(f):
            if (r["dataset"], r["scene"], r["model"], r["scorer"]) == ("prisma_bench", scene.name, "v0.2.0",
                                                                         "residual_mahalanobis(d=30,w=21x61)"):
                want = float(r["pauc_0.01"])
    for p in sorted((STAGE3 / "cache/scores/prisma_bench" / scene.name / V1_ID).glob("*.npz")):
        s = np.load(p)["score"].astype(np.float32)
        _, m = scene_metrics(s, scene.valid, scene.objects, {}, [METRICS[1]], [])
        if abs(m["pauc_0.01"] - want) < 1e-6:
            return s
    return None


def mnf_rx(name: str) -> np.ndarray:
    sys.path.insert(0, str(REPO / "research/experiments/hypersl-anomaly-bench"))
    from step2_evaluate import mnf_rx as f
    return f(name)


def main() -> None:
    import argparse
    from app.evaluation.scorers import score as eval_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="set1,set2,set3,set4")
    ap.add_argument("--tag", default="", help="suffix for the output files (partial runs)")
    a = ap.parse_args()
    scenes = a.scenes.split(",")
    rows, parts, hits, notes = {}, {}, {}, {}
    for scene in PrismaBench(str(ROOT), scenes):
        t0 = time.time()
        cls = np.load(FTH3_RUN / "cache/classes/prisma_bench" / f"{scene.name}__{FTH3_ID}.npz")["cls"]
        cloud = cls == CLASS_IDS["cloud"]
        maps = {REF: cached_rm(scene, FTH3_ID), "Indradhanu V2.0 (FT04)": cached_rm(scene, FT04_ID)}
        v1 = v1_rm(scene)
        if v1 is not None:
            maps["Indradhanu V1 (v0.2.0)"] = v1
        maps["PCA-RX (no model)"] = eval_score(SimpleNamespace(kind="pca_rx", spatial_sigma=0, d=30,
                                                               local_window=(21, 61)), scene.cube, None, scene.valid, None)
        maps["MNF-RX"] = mnf_rx(scene.name)
        for m in HAD_METHODS:
            p = OUT / scene.name / f"{m}.npz"
            if p.exists():
                maps[m] = np.load(p)["score"].astype(np.float32)
        for k in list(maps):
            maps[f"{k} | excl"] = np.where(cloud, 0.0, maps[k]).astype(np.float32)
        notes[scene.name] = {"cloud_px_valid": int((cloud & scene.valid).sum()), "methods": sorted(maps)}
        for label, s in maps.items():
            r, m = scene_metrics(s, scene.valid, scene.objects, scene.attrs, METRICS, ["tier"])
            rows.setdefault(label, {})[scene.name] = {"measures": m, "objects": r}
            parts.setdefault(label, {})[scene.name] = object_pauc_parts(s, scene.valid, scene.objects)
            for q in (1.0, 0.1):
                hit = object_hits(s, scene.valid, scene.objects, q)
                hits.setdefault((label, q), {})[scene.name] = {i: int(i in hit) for i in parts[label][scene.name]}
        print(scene.name, round(time.time() - t0, 1), "s", len(maps), "maps", flush=True)

    complete = {l: v for l, v in rows.items() if len(v) == len(scenes)}
    summary = {}
    for label, by_scene in complete.items():
        objs = [o for v in by_scene.values() for o in v["objects"]]
        sc = [v["measures"] for v in by_scene.values()]
        summary[label] = {
            "objects": len(objs), "caught_1_n": int(sum(o["hit_1"] for o in objs)),
            "caught_0.1_n": int(sum(o["hit_0.1"] for o in objs)),
            "caught_1": float(np.mean([o["hit_1"] for o in objs])),
            "caught_0.1": float(np.mean([o["hit_0.1"] for o in objs])),
            "pauc_0.01": float(np.mean([s["pauc_0.01"] for s in sc])),
            "roc_auc": float(np.mean([s["roc_auc"] for s in sc])),
            "tpr_at_0.001": float(np.mean([s["tpr_at_0.001"] for s in sc])),
            "per_scene": {n: {k: v["measures"][k] for k in ("pauc_0.01", "roc_auc", "tpr_at_0.001")} |
                          {"caught_1_n": int(sum(o["hit_1"] for o in v["objects"])),
                           "caught_0.1_n": int(sum(o["hit_0.1"] for o in v["objects"]))}
                          for n, v in by_scene.items()},
            "tier_caught_1": {t: float(np.mean([o["hit_1"] for o in objs if o.get("tier") == t]))
                              for t in ("easy", "medium", "subtle")},
            "chance_1_n": float(sum(o["chance_1"] for o in objs)),
            "chance_0.1_n": float(sum(o["chance_0.1"] for o in objs)),
        }
    boot = {}
    for label in complete:
        if label.replace(" | excl", "") == REF:
            continue
        ref = REF + " | excl" if label.endswith(" | excl") else REF
        if True:  # noqa: SIM108
            b1 = paired_bootstrap(parts[ref], parts[label], hits[(ref, 1.0)], hits[(label, 1.0)], n=2000, seed=0)
            b01 = paired_bootstrap(parts[ref], parts[label], hits[(ref, 0.1)], hits[(label, 0.1)], n=2000, seed=0)
            boot[label] = {"vs": ref, "pauc_diff": b1["pauc_diff"], "pauc_ci": list(b1["pauc_ci"]),
                           "pauc_p_b_better": b1["pauc_p_b_better"],
                           "caught_1_diff": b1["caught_diff"], "caught_1_ci": list(b1["caught_ci"]),
                           "caught_0.1_diff": b01["caught_diff"], "caught_0.1_ci": list(b01["caught_ci"])}
    # Pooled pAUC intervals for the chart: bootstrap of each method against itself gives its own percentile interval.
    own = {}
    for label in complete:
        rng = np.random.default_rng(0)
        P = parts[label]
        sc = sorted(P)
        arr = {s: np.array([P[s][i] for i in sorted(P[s])]) for s in sc}
        H1 = {s: np.array([hits[(label, 1.0)][s][i] for i in sorted(P[s])]) for s in sc}
        pa, ca = [], []
        for _ in range(2000):
            ks = {s: rng.integers(len(arr[s]), size=len(arr[s])) for s in sc}
            pa.append(np.mean([arr[s][ks[s], 0].sum() / arr[s][ks[s], 1].sum() for s in sc]))
            ca.append(sum(H1[s][ks[s]].sum() for s in sc))
        own[label] = {"pauc_ci": list(np.percentile(pa, [2.5, 97.5])), "caught_1_n_ci": list(np.percentile(ca, [2.5, 97.5]))}
    res = {"summary": summary, "bootstrap_vs_v2heavy": boot, "own_intervals": own, "notes": notes,
           "incomplete": sorted(set(rows) - set(complete))}
    (HERE / f"results{a.tag}.json").write_text(json.dumps(res, indent=1))
    with open(HERE / f"per_scene{a.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "cloud", "scene", "pauc_0.01", "roc_auc", "tpr_at_0.001", "caught_1_n", "caught_0.1_n"])
        for label, by_scene in rows.items():
            for n, v in by_scene.items():
                m = v["measures"]
                w.writerow([label.replace(" | excl", ""), "excluded" if label.endswith(" | excl") else "all", n,
                            round(m["pauc_0.01"], 4), round(m["roc_auc"], 4), round(m["tpr_at_0.001"], 4),
                            sum(o["hit_1"] for o in v["objects"]), sum(o["hit_0.1"] for o in v["objects"])])
    print(json.dumps({k: (round(v["pauc_0.01"], 4), v["caught_1_n"]) for k, v in summary.items()}, indent=0))


if __name__ == "__main__":
    main()
