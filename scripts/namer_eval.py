"""
Re-evaluate a namer checkpoint keeping per-sample predictions: bootstrap intervals and the open-set abstention rule.

    python scripts/namer_eval.py --ckpt runs/N002/ep40.pt --out runs/N002/eval_ep40 [--fills 0.1 0.25 0.5 1.0]
        [--batches 40] [--holdout textile_fabric]

Same protocol as train.py's evaluation (test shards, held-out spectra, own spectrum left out, fixed fill, man-made /
natural planted half and half). Per sensor and fill it saves samples.npz:
- the level-1 logit and the truth;
- the class, the predicted class and P(class | man-made);
- the best cosine to any man-made library entry other than the sample's own;
- whether the class is the held-out one.
Then it writes summary.json (intervals resample library spectra, not pixels) with:
- **Level 1:** recall at ACE's natural false-alarm rate, with a 95% bootstrap interval (1000 resamples of samples).
- **Level 2:** class accuracy, with a 95% bootstrap interval.
- **Open set, with a holdout:** the share of the held-out class called man-made, and the share given a confident
  (> 0.5) wrong class - before and after abstention. Abstention reports "man-made, class unknown" when the best
  man-made cosine (or the top P(class)) falls below a threshold. The threshold is set on the KNOWN classes' held-out
  spectra so that 90% / 95% of their man-made calls keep a class, then applied unchanged to the unseen class.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.spectral_match.namer.classes import MANMADE_CLASSES  # noqa: E402
from app.spectral_match.namer.generator import plant  # noqa: E402
from app.spectral_match.namer.train import NamerRun, _loader, decode  # noqa: E402

C = len(MANMADE_CLASSES)


def collect(run: NamerRun, src: dict, fill: float, batches: int) -> dict:
    model, cfg, dev = run.model, run.cfg, run.device
    g = torch.Generator(device=dev); g.manual_seed(0)
    out = {k: [] for k in ("logit", "manmade", "cls", "pred", "pcls", "best_cos", "held", "entry")}
    for i, batch in enumerate(_loader(src, "test", cfg)):
        if i >= batches:
            break
        pixels, valid, wl = decode(batch, dev)
        bank, wl = run.bank(src["name"], wl)
        if pixels.shape[1] != len(wl):
            continue
        keep = valid.float().mean(dim=(0, 2, 3)) > 0.5
        pixels, valid, wl, bank = pixels[:, keep], valid[:, keep], wl[keep], bank.bands(keep)
        s = plant(pixels, valid, bank, cfg["pixels_per_tile"], cfg["a_min"], cfg["gain"], (0.5, 0.5, 0.0), "test",
                  fill, g)
        if s is None:
            continue
        with torch.no_grad(), run._amp():
            z, logit, _ = model.forward_pixel(s["x"], s["r"], wl)
            e = model.forward_library(bank.refl, wl)
        z, e, logit = z.float(), e.float(), logit.float()
        train_lib = bank.manmade & ~bank.test
        if bank.holdout_class >= 0:
            train_lib &= bank.cls != bank.holdout_class
        lib_cls = torch.where(train_lib, bank.cls, torch.full_like(bank.cls, -1))
        if cfg.get("level1", "head") == "matching":
            lib_cls = torch.where(bank.manmade, lib_cls, torch.full_like(lib_cls, C))
            sc = model.class_scores(z, e, lib_cls, C + 1, exclude=s["entry"])
            logit, sc = model.level1_logit(sc), sc[:, :C]
        else:
            sc = model.class_scores(z, e, lib_cls, C, exclude=s["entry"])
        p = sc.softmax(1)
        cos = z @ e.T
        cos[:, ~train_lib] = -2.0                                         # library the model may name from
        rows = torch.nonzero(s["entry"] >= 0).squeeze(1)
        cos[rows, s["entry"][rows]] = -2.0                                # own spectrum left out
        for k, v in (("logit", logit), ("manmade", s["manmade"]), ("cls", s["cls"]), ("pred", p.argmax(1)),
                     ("pcls", p.max(1).values), ("best_cos", cos.max(1).values),
                     ("held", (s["cls"] == bank.holdout_class) & (s["cls"] >= 0)), ("entry", s["entry"])):
            out[k].append(v.detach().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def recall_at_fa(d, idx, fa):
    nat = d["logit"][idx][d["manmade"][idx] < 0.5]
    mm = d["logit"][idx][(d["manmade"][idx] > 0.5) & ~d["held"][idx]]
    if not len(nat) or not len(mm):
        return float("nan")
    return float((mm > np.quantile(nat, 1 - fa)).mean())


def boot(fn, entry, reps=500, seed=0):
    """95% interval, resampling library SPECTRA (clusters), not pixels: the pixels planted from one spectrum are not
    independent, and the unseen-spectrum set (~34) is the real sample size. A pixel-level bootstrap gave ~+/-0.003."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(entry == e) for e in np.unique(entry)]
    vals = [fn(np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(reps)]
    return [round(float(np.nanpercentile(vals, 2.5)), 4), round(float(np.nanpercentile(vals, 97.5)), 4)]


def summarise(d, fa):
    n = len(d["logit"])
    allidx = np.arange(n)
    known = (d["manmade"] > 0.5) & ~d["held"]
    l2 = lambda idx: float((d["pred"][idx] == d["cls"][idx])[known[idx]].mean()) if known[idx].any() else float("nan")
    r = {"n": int(n), "n_spectra": int(len(np.unique(d["entry"]))), "l2_acc": round(l2(allidx), 4),
         "l2_acc_ci": boot(l2, d["entry"])}
    if fa is not None:
        r["l1_recall_at_ace_fa"] = round(recall_at_fa(d, allidx, fa), 4)
        r["l1_recall_ci"] = boot(lambda idx: recall_at_fa(d, idx, fa), d["entry"])
    held = d["held"]
    if held.any():
        mm_call = d["logit"] > 0
        r["open_called_manmade"] = round(float(mm_call[held].mean()), 4)
        r["open_confident_wrong"] = round(float((mm_call & (d["pcls"] > 0.5))[held].mean()), 4)
        kn_mm = known & mm_call
        for sig in ("best_cos", "pcls"):
            for keep in (0.90, 0.95):
                t = float(np.quantile(d[sig][kn_mm], 1 - keep))                 # known classes keep `keep`
                named = d[sig] >= t
                r[f"abstain_{sig}_{int(keep * 100)}"] = {
                    "threshold": round(t, 4),
                    "known_kept": round(float(named[kn_mm].mean()), 4),
                    "known_acc_when_named": round(float((d["pred"] == d["cls"])[kn_mm & named].mean()), 4),
                    "open_confident_wrong": round(float((mm_call & named & (d["pcls"] > 0.5))[held].mean()), 4),
                    "open_unknown": round(float((mm_call & ~named)[held].mean()), 4)}
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fills", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--holdout")
    ap.add_argument("--config", help="override the checkpoint's config (paths) with this JSON")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    if a.config:
        cfg.update(json.loads(Path(a.config).read_text()))
    if a.holdout:
        cfg["holdout_class"] = a.holdout
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg["run_dir"] = str(out / "_run")                                     # NamerRun writes nothing here
    run = NamerRun(cfg)
    run.model.load_state_dict(ck["model"])
    run.model.eval()
    summary = {"ckpt": a.ckpt, "epoch": ck["epoch"] + 1, "holdout": cfg.get("holdout_class")}
    for src in cfg["sources"]:
        for f in a.fills:
            d = collect(run, src, f, a.batches)
            np.savez_compressed(out / f"samples_{src['name']}_{f}.npz", **d)
            summary[f"{src['name']}_{f}"] = summarise(d, run._ace_false_alarm(src["name"], f))
            print(src["name"], f, json.dumps(summary[f"{src['name']}_{f}"]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
