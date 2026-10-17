"""
Run an evaluation (docs/lld/evaluation.md): every dataset x model x scorer of a config, every metric, a report.

    python scripts/evaluate.py configs/eval/<name>.json [--output <dir>]

Writes per_object.csv, per_scene.csv, summary.json, report.md, config.json and provenance.json to the output
directory; score maps are cached there, so re-running with new metrics or groups re-runs no model.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import Evaluation, EvaluationConfig  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    p.add_argument("config")
    p.add_argument("--output", help="override the config's output directory")
    a = p.parse_args()
    cfg = EvaluationConfig.from_file(a.config)
    if a.output:
        cfg = cfg.model_copy(update={"output": a.output})
    Evaluation(cfg).run()
    print(f"report: {Path(cfg.output) / 'report.md'}")


if __name__ == "__main__":
    main()
