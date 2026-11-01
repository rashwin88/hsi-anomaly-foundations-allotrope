"""
Step 2 of the HyperSL anomaly bench: measure HyperSL's score maps (step 1) with the evaluator's own measures, beside
Indradhanu FTH3 epoch 5 (the release candidate) and two classical detectors, on the same 909 objects.

Runs on the Windows product venv:  C:/av/Scripts/python.exe step2_evaluate.py
Reads  OUT/<set>_scores.npz (step 1), the FTH3 evaluation's cached score and class maps (C:/es/eval/fth2_ftb2,
       written by scripts/evaluate.py configs/eval/fth3.json on 2026-10-29),
and computes PCA-RX (the evaluator's pca_rx scorer) and MNF-RX (the product's MNFCompressionDetector, 10 components)
on the bench cube. Every method is scored plain and with cloud excluded by FTH3 epoch 5's thick-cloud class map (the
release candidate's rule, as in the FTH3 report).

Measures: app.evaluation.metrics.scene_metrics (pAUC at FPR <= 0.01, ROC-AUC, TPR at FPR 0.001, objects caught in
the top 1% / 0.1%), summarised as app.evaluation.report.summarise does (pixel measures: mean over scenes; caught: share
of all 909 objects). Paired object bootstrap (app.evaluation.significance) of each method against FTH3.
Writes results.json and results.md beside this file.
"""
from __future__ import annotations

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
from app.evaluation.datasets import PRISMA_SCENES, PRODUCT_EXCLUSIONS, PrismaBench  # noqa: E402
from app.evaluation.metrics import object_hits, scene_metrics  # noqa: E402
from app.evaluation.significance import object_pauc_parts, paired_bootstrap  # noqa: E402

ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT = ROOT / "hypersl_anomaly_bench"
FTH3_RUN = Path(r"C:\es\eval\fth2_ftb2")
FTH3_ID = "FTH3_ep5-fefed77b7a"
METRICS = [SimpleNamespace(kind="roc_auc"), SimpleNamespace(kind="pauc", max_fpr=0.01),
           SimpleNamespace(kind="tpr_at", fpr=0.001), SimpleNamespace(kind="objects_caught", top_pct=[1.0, 0.1])]


def fth3_maps(scene) -> dict:
    """FTH3 epoch 5's cached RM and 'excl. FTH3_ep5 thick' score maps, found by the runner's own cache key."""
    cfg = EvaluationConfig.from_file(REPO / "configs/eval/fth3.json")
    he5 = ROOT / PRISMA_SCENES[scene.name][0]
    fp = f"{he5.stat().st_size}-{he5.stat().st_mtime_ns}"
    out = {}
    for spec in cfg.scorers:
        if spec.label not in ("RM", "excl. FTH3_ep5 thick"):
            continue
        key = hashlib.sha1((spec.model_dump_json() + "|" + fp).encode()).hexdigest()[:10]
        p = FTH3_RUN / "cache/scores/prisma_bench" / scene.name / FTH3_ID / f"{key}.npz"
        out[spec.label] = np.load(p)["score"].astype(np.float32)
    return out


def mnf_rx(name: str) -> np.ndarray:
    """The product's MNF-RX (10 components) on the scene's native vendable, NaN -> 0."""
    from app.detectors.mnf_compression_detector import MNFCompressionDetector
    from app.models.dataset.vendables import BandFilterConfig
    from app.models.file_processing.sources import FileSourceConfig
    from app.utils.dataset_builder.prisma_dataset_builder import PrismaDatasetBuilder
    cfg = BandFilterConfig(exclusion_ranges=PRODUCT_EXCLUSIONS, edge_bands_to_trim=3, min_valid_pixel_pct=20.0,
                           max_invalid_voxel_fraction=0.4, quality_masks_to_apply=[], common_wavelength_grid=None)
    v = PrismaDatasetBuilder(FileSourceConfig(source_path=str(ROOT / PRISMA_SCENES[name][0]))).vend_dataset(
        band_filter_config=cfg)
    det = MNFCompressionDetector(v)
    det.fit(n_components=10, exclusion_ranges=PRODUCT_EXCLUSIONS)
    return np.nan_to_num(det.detect(v.normalized_hyperspectral_cube), nan=0.0).astype(np.float32)


def main() -> None:
    from app.evaluation.scorers import score as eval_score
    rows, parts, hits, timing = {}, {}, {}, {}
    for scene in PrismaBench(str(ROOT)):
        t0 = time.time()
        cls = np.load(FTH3_RUN / "cache/classes/prisma_bench" / f"{scene.name}__{FTH3_ID}.npz")["cls"]
        cloud = cls == CLASS_IDS["cloud"]
        h = np.load(OUT / f"{scene.name}_scores.npz")
        f3 = fth3_maps(scene)
        t1 = time.time()
        pca = eval_score(SimpleNamespace(kind="pca_rx", spatial_sigma=0, d=30, local_window=(21, 61)),
                         scene.cube, None, scene.valid, None)
        timing[scene.name] = {"pca_rx_s": round(time.time() - t1, 1)}
        t1 = time.time()
        mnf = mnf_rx(scene.name)
        timing[scene.name]["mnf_rx_s"] = round(time.time() - t1, 1)
        maps = {"FTH3_ep5 RM": f3["RM"], "FTH3_ep5 RM, cloud excl.": f3["excl. FTH3_ep5 thick"],
                "HyperSL L2": h["hsl_l2"], "HyperSL loss": h["hsl_loss"], "HyperSL RM": h["hsl_rm"],
                "PCA-RX": pca, "MNF-RX": mnf}
        for k in ("HyperSL L2", "HyperSL loss", "HyperSL RM", "PCA-RX", "MNF-RX"):
            maps[f"{k}, cloud excl."] = np.where(cloud, 0.0, maps[k]).astype(np.float32)
        check = np.abs(maps["FTH3_ep5 RM, cloud excl."] - np.where(cloud, 0.0, maps["FTH3_ep5 RM"])).max()
        timing[scene.name]["fth3_excl_vs_mask_maxdiff"] = float(check)
        timing[scene.name]["cloud_px_valid"] = int((cloud & scene.valid).sum())
        for label, s in maps.items():
            r, m = scene_metrics(s, scene.valid, scene.objects, scene.attrs, METRICS, ["tier"])
            rows.setdefault(label, {})[scene.name] = {"measures": m, "objects": r}
            parts.setdefault(label, {})[scene.name] = object_pauc_parts(s, scene.valid, scene.objects)
            hit = object_hits(s, scene.valid, scene.objects, 1.0)
            hits.setdefault(label, {})[scene.name] = {i: int(i in hit) for i in parts[label][scene.name]}
        timing[scene.name]["eval_s"] = round(time.time() - t0, 1)
        print(scene.name, timing[scene.name], flush=True)

    summary = {}
    for label, by_scene in rows.items():
        objs = [o for v in by_scene.values() for o in v["objects"]]
        sc = [v["measures"] for v in by_scene.values()]
        summary[label] = {
            "objects": len(objs),
            "caught_1": float(np.mean([o["hit_1"] for o in objs])),
            "caught_0.1": float(np.mean([o["hit_0.1"] for o in objs])),
            "caught_1_n": int(sum(o["hit_1"] for o in objs)),
            "pauc_0.01": float(np.mean([s["pauc_0.01"] for s in sc])),
            "roc_auc": float(np.mean([s["roc_auc"] for s in sc])),
            "tpr_at_0.001": float(np.mean([s["tpr_at_0.001"] for s in sc])),
            "per_scene_pauc": {n: v["measures"]["pauc_0.01"] for n, v in by_scene.items()},
            "per_scene_caught_1": {n: int(sum(o["hit_1"] for o in v["objects"])) for n, v in by_scene.items()},
            "tier_caught_1": {t: float(np.mean([o["hit_1"] for o in objs if o.get("tier") == t]))
                              for t in ("easy", "medium", "subtle")},
            "chance_1": float(np.mean([o["chance_1"] for o in objs])),
        }
    boot = {}
    for ref in ("FTH3_ep5 RM", "FTH3_ep5 RM, cloud excl."):
        excl = ref.endswith("excl.")
        for label in rows:
            if label.startswith("FTH3") or label.endswith("excl.") != excl:
                continue
            b = paired_bootstrap(parts[ref], parts[label], hits[ref], hits[label], n=2000, seed=0)
            boot[f"{label} - {ref}"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}
    # The self-completion test: the same scorer (residual Mahalanobis) on HyperSL's residual against on the cube itself
    # (PCA-RX = residual Mahalanobis with a zero reconstruction). A reconstruction that helps must beat "no model".
    for suffix in ("", ", cloud excl."):
        ref, label = f"PCA-RX{suffix}", f"HyperSL RM{suffix}"
        b = paired_bootstrap(parts[ref], parts[label], hits[ref], hits[label], n=2000, seed=0)
        boot[f"{label} - {ref}"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in b.items()}
    res = {"summary": summary, "bootstrap": boot, "timing": timing,
           "step1": json.loads((OUT / "step1_log.json").read_text())}
    (HERE / "results.json").write_text(json.dumps(res, indent=1))

    lines = ["| method | caught top 1% | caught top 0.1% | pAUC (FPR<=0.01) | ROC-AUC | TPR at 0.001 |",
             "|---|---|---|---|---|---|"]
    for label, s in summary.items():
        lines.append(f"| {label} | {s['caught_1_n']} ({100 * s['caught_1']:.1f}%) | {100 * s['caught_0.1']:.1f}% | "
                     f"{s['pauc_0.01']:.4f} | {s['roc_auc']:.3f} | {s['tpr_at_0.001']:.3f} |")
    lines += ["", "| difference (B - A) | pAUC diff [95% CI] | caught top 1% diff [95% CI] |", "|---|---|---|"]
    for k, b in boot.items():
        lines.append(f"| {k} | {b['pauc_diff']:+.4f} [{b['pauc_ci'][0]:+.4f}, {b['pauc_ci'][1]:+.4f}] | "
                     f"{100 * b['caught_diff']:+.1f} [{100 * b['caught_ci'][0]:+.1f}, {100 * b['caught_ci'][1]:+.1f}] pts |")
    (HERE / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
