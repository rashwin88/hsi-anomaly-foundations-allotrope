"""Where does the uncertainty head put its expected error? Percentile of b at target pixels vs the scene, and target
pixels' b vs their 3-8 px surroundings - epoch 1 vs epoch 3."""
import sys, numpy as np
from scipy import ndimage
from app.evaluation.config import ModelSpec
from app.evaluation.datasets import PrismaBench
from app.evaluation.models import EvalModel
from pathlib import Path
for ck in ["C:/es/models/FT06_ep1.pt", "C:/es/models/FT06_best.pt"]:
    m = EvalModel(ModelSpec(name=Path(ck).stem, checkpoint=ck), Path("C:/es/eval/cache_b"))
    for sc in PrismaBench("C:/Users/FITT/Documents/stage1_datasets", ["set1", "set2", "set4"]):
        _, _, valid, lb = m.reconstruct(sc)
        t = (sc.objects > 0) & valid
        ring = ndimage.binary_dilation(t, iterations=8) & ~ndimage.binary_dilation(t, iterations=3) & valid
        pct = (np.argsort(np.argsort(lb[valid])) / valid.sum())
        P = np.zeros(lb.shape); P[valid] = pct
        print(m.name, sc.name, f"target b pct median {np.median(P[t]):.2f}  ring {np.median(P[ring]):.2f}  "
              f"target/ring b ratio {np.exp(np.median(lb[t]) - np.median(lb[ring])):.2f}", flush=True)
    m.release()
