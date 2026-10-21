"""Paired object bootstrap (app/evaluation/significance.py)."""
import numpy as np

from app.evaluation.metrics import exact_roc, pauc
from app.evaluation.significance import object_pauc_parts, paired_bootstrap


def scene(seed):
    rng = np.random.default_rng(seed)
    score = rng.random((60, 60))
    objects = np.zeros((60, 60), int)
    for k in range(1, 13):
        y, x = rng.integers(0, 58, 2)
        objects[y:y + 2, x:x + 2] = k
        score[y:y + 2, x:x + 2] += rng.random() * 0.5          # targets somewhat higher
    return score, np.ones((60, 60), bool), objects


def test_object_parts_reproduce_pauc_without_ties():
    score, valid, objects = scene(0)
    parts = object_pauc_parts(score, valid, objects, max_fpr=0.05)
    from_parts = sum(c for c, _ in parts.values()) / sum(n for _, n in parts.values())
    fpr, tpr = exact_roc(score[valid], objects[valid] > 0)
    assert abs(from_parts - pauc(fpr, tpr, 0.05)) < 1e-9


def test_identical_scorers_differ_by_exactly_zero_and_a_better_one_wins():
    score, valid, objects = scene(1)
    a = {"s": object_pauc_parts(score, valid, objects)}
    r = paired_bootstrap(a, a, n=200)
    assert r["pauc_diff"] == 0 and r["pauc_ci"] == (0.0, 0.0)
    better = score + (objects > 0) * 1.0                          # every target lifted clear of the background
    r = paired_bootstrap(a, {"s": object_pauc_parts(better, valid, objects)}, n=200)
    assert r["pauc_ci"][0] > 0 and r["pauc_p_b_better"] == 1.0
