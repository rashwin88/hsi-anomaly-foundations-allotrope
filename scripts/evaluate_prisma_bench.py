"""
Score Indradhanu-family checkpoints on the 4 labelled PRISMA anomaly scenes.

    python scripts/evaluate_prisma_bench.py --bench-root C:/Users/FITT/Documents/stage1_datasets \
        --baseline C:/es/models/hyperspectral_segformer_mae_v0.2.0_epoch200.pt \
        --model E004=C:/es/models/E004_best.pt@se202,native --out eval.json

Every model is run through the product's own path: the SegFormer full-scene inferencer (128 px
patches, stride 32, two complementary token masks, erosion 15, overlap averaging), then the
product's compute_score (L1, SAM, combined with alpha 0.5) and compute_roc. The baseline reads the
scene exactly as band_filter_apply prepares it (product165); a spectral-stem checkpoint reads it in
the modes listed after `@` (se202 by default - see app/utils/evaluation/prisma_bench.py).

Prints one table (combined ROC-AUC per scene and the mean - the decision-rule number) plus the L1
and SAM AUCs, TPR at 1% FPR and objects caught at the top-1% cut; writes everything to --out.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.foundation_models.inferencers.hyperspectral_segformer_mae_inferencer import (  # noqa: E402
    HyperspectralSegFormerMAEInferencer)
from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer  # noqa: E402
from app.models.training.inference_config import InferenceConfig  # noqa: E402
from app.utils.anomaly_detection.scoring import compute_score  # noqa: E402
from app.utils.evaluation.prisma_bench import SCENES, load_scene, score_metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
# The product's Indradhanu defaults (backend/allotrope/foundation_models/resolver.py).
PATCH, STRIDE, BATCH, ALPHA = 128, 32, 8, 0.5
METHODS = ("combined", "L1", "SAM")


def build(name: str, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mc = ck["config"]["model_config"]
    common = dict(model_config=mc, checkpoint_path=ckpt_path, patch_size=PATCH, stride=STRIDE,
                  inference_batch_size=BATCH)
    if mc["model_type"] == "hyperspectral_segformer_mae":
        return HyperspectralSegFormerMAEInferencer(InferenceConfig(
            foundation_model_name="hyperspectral_segformer_mae", **common,
            pixel_stats_path=str(REPO / "app" / "constants" / "hyperspectral.json")))
    return SpectralStemInferencer(InferenceConfig(foundation_model_name="spectral_stem_mae", **common))


def evaluate(inferencer, scene) -> dict:
    if isinstance(inferencer, SpectralStemInferencer):
        inferencer.set_wavelengths(scene.wavelengths_nm)
    t = time.time()
    with torch.no_grad():
        recon = inferencer.predict_full_scene(torch.from_numpy(scene.cube), torch.from_numpy(scene.spatial_valid[None]).float())
    recon = recon.detach().cpu().numpy().astype(np.float32)
    out = {"seconds": round(time.time() - t, 1)}
    for method in METHODS:
        score = compute_score(scene.cube, recon, scene.spatial_valid, method=method, combined_weight=ALPHA)
        out[method] = score_metrics(score, scene.gt, scene.spatial_valid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--baseline", help="production Indradhanu checkpoint (read in product165 mode)")
    ap.add_argument("--model", action="append", default=[], help="NAME=checkpoint[@mode,mode] (default mode se202)")
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--out", default="prisma_bench.json")
    args = ap.parse_args()

    runs = []                                            # (label, checkpoint, mode)
    if args.baseline:
        runs.append(("v0.2.0", args.baseline, "product165"))
    for spec in args.model:
        name, _, rest = spec.partition("=")
        path, _, modes = rest.partition("@")
        for mode in (modes or "se202").split(","):
            runs.append((f"{name}/{mode}", path, mode))

    results: dict = {}
    for label, path, mode in runs:
        inferencer = build(label, path)
        for name in args.scenes.split(","):
            scene = load_scene(args.bench_root, name, mode)
            results.setdefault(label, {})[name] = r = evaluate(inferencer, scene)
            print(f"{label:22s} {name}: combined AUC {r['combined']['auc']:.3f}  L1 {r['L1']['auc']:.3f}  "
                  f"SAM {r['SAM']['auc']:.3f}  caught {r['combined']['objects_caught']}/{r['combined']['objects']}  "
                  f"({r['seconds']} s)", flush=True)
        del inferencer
        torch.cuda.empty_cache()

    scenes = args.scenes.split(",")
    print("\ncombined ROC-AUC (product scoring, alpha 0.5)")
    print(f"{'model':22s} " + " ".join(f"{s:>6s}" for s in scenes) + "    mean")
    for label, per in results.items():
        aucs = [per[s]["combined"]["auc"] for s in scenes]
        print(f"{label:22s} " + " ".join(f"{a:6.3f}" for a in aucs) + f"   {np.mean(aucs):.3f}")
        per["mean_combined_auc"] = float(np.mean(aucs))
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
