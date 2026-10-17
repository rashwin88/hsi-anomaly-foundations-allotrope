"""Evaluation of anomaly-detection models: datasets x models x scorers x metrics from one config.
docs/lld/evaluation.md; run with scripts/evaluate.py <config.json>."""

from app.evaluation.config import EvaluationConfig
from app.evaluation.runner import Evaluation

__all__ = ["Evaluation", "EvaluationConfig"]
