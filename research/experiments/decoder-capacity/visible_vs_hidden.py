"""Decoder capacity or unpredictability? FT04's error on pixels it can SEE (no MAE mask - pure autoencoding) vs on
pixels it must PREDICT (the product's two complementary checkerboard passes). If visible-pixel error is close to
hidden-pixel error, the decoder cannot even reproduce what it sees - capacity is the limit. If it is much lower, the
error is what the context cannot predict. Mean |r| per band in normalised units, PRISMA bench tiles, no training."""
import sys
import numpy as np, torch
from pathlib import Path
from app.evaluation.config import ModelSpec
from app.evaluation.datasets import PrismaBench
from app.evaluation.models import EvalModel

# usage: visible_vs_hidden.py [--checkpoint path] [scenes...]
args = sys.argv[1:]
ckpt = "C:/es/models/FT04_best.pt"
if args[:1] == ["--checkpoint"]:
    ckpt, args = args[1], args[2:]
m = EvalModel(ModelSpec(name=Path(ckpt).stem, checkpoint=ckpt), Path("C:/es/eval/cache_b"))
inf = m.inferencer
net = inf.model.inner.eval()
T = 128
for sc in PrismaBench("C:/Users/FITT/Documents/stage1_datasets", args or ["set1", "set2", "set3", "set4"]):
    cube, wl, valid = m._read(sc, m.mode)
    w = torch.as_tensor(wl, dtype=torch.float32, device=inf.device)
    rng = np.random.default_rng(0)
    H, W = valid.shape
    errs = {"visible (no mask)": [], "hidden (checkerboard)": []}
    tgt = {"visible (no mask)": [], "hidden (checkerboard)": []}
    n = 0
    while n < 24:
        y, x = rng.integers(0, H - T), rng.integers(0, W - T)
        v = valid[y:y + T, x:x + T]
        if v.mean() < 0.95:
            continue
        n += 1
        xt = torch.from_numpy(np.ascontiguousarray(cube[:, y:y + T, x:x + T]))[None].to(inf.device)
        vt = torch.from_numpy(v)[None, None].to(inf.device)
        tg = sc.objects[y:y + T, x:x + T] > 0
        with torch.no_grad():
            z = net.stats.normalize(xt, w)
            full = net.stats.normalize(net(xt, w, vt.expand(-1, len(wl), -1, -1)), w)
            hid = net.stats.normalize(inf.predict(xt, vt.float()), w)       # the product's two-pass, every pixel hidden once
        for k, r in (("visible (no mask)", full), ("hidden (checkerboard)", hid)):
            e = (r - z).abs().mean(1)[0].cpu().numpy()
            errs[k].append(e[v & ~tg]); tgt[k].append(e[v & tg])
    s = " | ".join(f"{k}: bg {np.concatenate(errs[k]).mean():.3f}" + (f", targets {np.concatenate(tgt[k]).mean():.3f}"
                   if sum(len(a) for a in tgt[k]) else "") for k in errs)
    print(sc.name, s, flush=True)
