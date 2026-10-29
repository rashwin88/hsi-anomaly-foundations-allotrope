"""Product path for Indradhanu V2 = Heavy (resolver manifest -> get_inferencer -> read_bands -> predict ->
compute_score(residual_mahalanobis) -> segment_full_scene(thin_cloud=capability) -> zero cloud) vs the evaluator's
FTH3_ep5 "excl. FTH3_ep5 thick" numbers on the PRISMA bench (C:/es/eval/fth2_ftb2/per_scene.csv).
Adapted from ../FT04-final-joint/product_parity.py; the manifest is the one shipped as v2.1.0.

    PYTHONPATH=. PYTHONUTF8=1 C:/av/Scripts/python.exe \
        research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/product_parity.py set1 set4
"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "backend")
from allotrope.foundation_models.resolver import _from_manifest
from app.foundation_models.inferencers.inferencer_factory import get_inferencer
from app.models.training.inference_config import InferenceConfig
from app.utils.anomaly_detection.scoring import compute_score
from app.evaluation.datasets import PrismaBench
from app.evaluation.metrics import exact_roc, auc_exact, pauc

d = Path("C:/es/ship/fth3/spectral_stem_finetune")
m = _from_manifest(d, d / "current.json")
assert m.thin_cloud_is_cloud is False and m.exclude_classes == ("cloud",)
inf = get_inferencer(InferenceConfig(foundation_model_name=m.foundation_model_name, model_config=m.model_config,
    checkpoint_path=m.checkpoint_abs_path, patch_size=m.default_patch_size, stride=m.default_stride,
    inference_batch_size=m.default_batch_size, pixel_stats_path=m.pixel_stats_abs_path))
for sc in PrismaBench("C:/Users/FITT/Documents/stage1_datasets", sys.argv[1:]):
    cube, wl, valid = sc.read("native")
    cube_m, wl_m, _ = inf.read_bands(cube, wl)
    with torch.no_grad():
        recon = inf.predict_full_scene(torch.from_numpy(np.ascontiguousarray(cube_m)).float(),
                                       torch.from_numpy(valid[None].astype(np.float32))).cpu().numpy()
    score = compute_score(cube_m.astype(np.float32), recon.astype(np.float32), valid.astype(np.uint8),
                          method="residual_mahalanobis")
    cls, _ = inf.segment_full_scene(cube_m, valid, thin_cloud=m.thin_cloud_is_cloud)
    excl = np.isin(cls, [inf.CLASS_NAMES.index(c) for c in m.exclude_classes])
    for name, s in (("plain", score), ("thick-cloud-excluded", np.where(excl, 0.0, score))):
        fpr, tpr = exact_roc(s[valid], sc.objects[valid] > 0)
        print(sc.name, name, f"bands={len(wl_m)} excl={100*excl[valid].mean():.2f}%",
              f"roc={auc_exact(fpr, tpr):.4f} pauc={pauc(fpr, tpr):.4f}", flush=True)
