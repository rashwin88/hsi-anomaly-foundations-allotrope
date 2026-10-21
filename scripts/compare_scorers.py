"""
Is B better than A on the real bench? Paired object bootstrap from an evaluation's cached score maps
(app/evaluation/significance.py).

    python scripts/compare_scorers.py configs/eval/scene_adapt.json "FT04_ep10|RM, cloud excluded" \
        "adapt50|RM, cloud excluded" [--top 0.1] [--boot 2000]

Each side is "<model name>|<scorer label>" as in the run's per_scene.csv; `-` as the model name for model-free
scorers. Only prisma_bench scenes; the run must have finished (score maps and per_object.csv are read, nothing is
recomputed).
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.config import EvaluationConfig      # noqa: E402
from app.evaluation.datasets import make_dataset         # noqa: E402
from app.evaluation.models import EvalModel              # noqa: E402
from app.evaluation.runner import Evaluation             # noqa: E402
from app.evaluation.significance import object_pauc_parts, paired_bootstrap   # noqa: E402

import numpy as np                                       # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--top", type=float, default=0.1, help="objects-caught cut-off, %% of pixels (1.0 or 0.1)")
    p.add_argument("--boot", type=int, default=2000)
    args = p.parse_args()
    cfg = EvaluationConfig.model_validate(json.loads(Path(args.config).read_text()))
    run = Evaluation(cfg)
    sides = [s.split("|", 1) for s in (args.a, args.b)]
    models = {m.name: EvalModel(m, run.cache) for m in cfg.models if m.name in {n for n, _ in sides}}
    specs = {s.label: s for s in cfg.scorers}
    objs = pd.read_csv(Path(cfg.output) / "per_object.csv")
    parts, hits = ([{}, {}], [{}, {}])
    for ds in cfg.datasets:
        if ds.kind != "prisma_bench":
            continue
        for scene in make_dataset(ds):
            for k, (mname, label) in enumerate(sides):
                path = run._score_path(scene, models.get(mname), specs[label])
                score = np.load(path)["score"].astype(np.float32)
                parts[k][scene.name] = object_pauc_parts(score, scene.valid, scene.objects)
                rows = objs[(objs.scene == scene.name) & (objs.model == mname) & (objs.scorer == label)]
                hits[k][scene.name] = dict(zip(rows.object.astype(int), rows[f"hit_{args.top:g}"].astype(int)))
    r = paired_bootstrap(parts[0], parts[1], hits[0], hits[1], n=args.boot)
    print(f"A = {args.a}\nB = {args.b}")
    print(f"pAUC<=1%: A {r['pauc_a']:.4f}  B {r['pauc_b']:.4f}  B-A {r['pauc_diff']:+.4f}  "
          f"95% [{r['pauc_ci'][0]:+.4f}, {r['pauc_ci'][1]:+.4f}]  P(B>A) {r['pauc_p_b_better']:.3f}")
    print(f"caught top {args.top:g}%: A {100 * r['caught_a']:.1f}%  B {100 * r['caught_b']:.1f}%  "
          f"B-A {100 * r['caught_diff']:+.1f} pts  95% [{100 * r['caught_ci'][0]:+.1f}, {100 * r['caught_ci'][1]:+.1f}]")


if __name__ == "__main__":
    main()
