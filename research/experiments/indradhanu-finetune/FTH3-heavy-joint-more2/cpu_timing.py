"""Seconds per PRISMA scene through anomaly_scoring's foundation path on CPU, Heavy (v2.1.0, current.json) vs FT04
(v2.0.0, the manifest's first alternative): load, read_bands, predict_full_scene, compute_score, segment_full_scene.
Runs inside the worker container (docker exec -w /srv -e PYTHONPATH=/srv allotrope_worker python cpu_timing.py
<scene.npz>); the scene is set1's native cube, wavelengths and valid mask dumped with
`np.savez(path, cube=cube, wl=wl, valid=valid)` from app.evaluation.datasets.PrismaBench.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from allotrope.config import settings
from app.foundation_models.inferencers.inferencer_factory import get_inferencer
from app.models.training.inference_config import InferenceConfig
from app.utils.anomaly_detection.scoring import compute_score

d = Path(settings.models_dir) / "spectral_stem_finetune"
man = json.loads((d / "current.json").read_text(encoding="utf-8"))
z = np.load(sys.argv[1])
cube, wl, valid = z["cube"], z["wl"], z["valid"].astype(bool)
print(f"scene {cube.shape}, torch threads {torch.get_num_threads()}, device cpu", flush=True)
for name, cur in (("Heavy " + man["current"]["version"], man["current"]),
                  ("FT04 " + man["alternatives"][0]["version"], man["alternatives"][0])):
    t = {}
    t0 = time.time()
    inf = get_inferencer(InferenceConfig(foundation_model_name="spectral_stem_finetune", model_config=cur["model_config"],
        checkpoint_path=str(d / cur["file"]), patch_size=128, stride=32, inference_batch_size=8, device="cpu"))
    t["load"] = time.time() - t0; t0 = time.time()
    cube_m, _, _ = inf.read_bands(cube, wl)
    t["read_bands"] = time.time() - t0; t0 = time.time()
    with torch.no_grad():
        recon = inf.predict_full_scene(torch.from_numpy(np.ascontiguousarray(cube_m)).float(),
                                       torch.from_numpy(valid[None].astype(np.float32))).cpu().numpy()
    t["predict"] = time.time() - t0; t0 = time.time()
    compute_score(cube_m.astype(np.float32), recon.astype(np.float32), valid.astype(np.uint8),
                  method="residual_mahalanobis")
    t["score"] = time.time() - t0; t0 = time.time()
    inf.segment_full_scene(cube_m, valid, thin_cloud=False)
    t["segment"] = time.time() - t0
    print(name, " ".join(f"{k}={v:.1f}s" for k, v in t.items()), f"total={sum(t.values()):.1f}s", flush=True)
    del inf, recon
