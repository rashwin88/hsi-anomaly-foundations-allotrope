"""Speed and learning-rate probe for gated scene adaptation on PRISMA set1 - unsupervised signals only (the adaptation
loss on gated-in pixels), no ground truth."""
import time, numpy as np, torch
from pathlib import Path
from app.evaluation.config import ModelSpec
from app.evaluation.datasets import PrismaBench
from app.evaluation.models import EvalModel
from app.foundation_models.scene_adaptation import adapt, gate_mask
from app.utils.anomaly_detection.residual_mahalanobis import residual_mahalanobis
m = EvalModel(ModelSpec(name="FT04", checkpoint="C:/es/models/FT04_best.pt"), Path("C:/es/eval/cache_b"))
sc = next(iter(PrismaBench("C:/Users/FITT/Documents/stage1_datasets", ["set1"])))
cube, wl, valid = m._read(sc, m.mode)
x, mk = torch.from_numpy(cube), torch.from_numpy(valid[None].astype(np.float32))
with torch.no_grad():
    r0 = m.inferencer.predict_full_scene(x, mk).cpu().numpy()
gate = gate_mask(residual_mahalanobis(cube, r0, valid), valid, 1.0, 2)
print("gated px %.2f%%" % (100 * gate[valid].mean()), "bands", len(wl), flush=True)
net = m.inferencer.model.inner
orig = {k: v.clone() for k, v in net.state_dict().items()}
for lr in (1e-5, 3e-5, 1e-4):
    net.load_state_dict(orig)
    t = time.time()
    L = adapt(net, cube, wl, valid, gate, 300, lr=lr, device=m.inferencer.device)
    print(f"lr {lr:g}: {300/(time.time()-t):.1f} steps/s  loss first50 {np.mean(L[:50]):.4f} 100-150 {np.mean(L[100:150]):.4f} last50 {np.mean(L[-50:]):.4f}", flush=True)
