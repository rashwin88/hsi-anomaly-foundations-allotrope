"""app/evaluation/metrics.py - numbers worked by hand."""
from types import SimpleNamespace as M

import numpy as np
import pytest

from app.evaluation.metrics import auc_exact, chance_hit, exact_roc, pauc, scene_metrics, three_d_roc, tpr_at


def test_by_hand():
    # Targets at 0.9 and 0.7. ROC steps: (0, 1/2), (1/3, 1/2), (1/3, 1), (2/3, 1), (1, 1) -> AUC 1/3*1/2 + 2/3 = 0.8333.
    # Normalised scores 1, .889, .778, .111, 0: targets mean .889, background mean (.889 + .111 + 0) / 3 = .333.
    s, g = np.array([0.9, 0.8, 0.7, 0.1, 0.0]), np.array([1, 0, 1, 0, 0], bool)
    fpr, tpr = exact_roc(s, g)
    auc = auc_exact(fpr, tpr)
    assert auc == pytest.approx(5 / 6)
    assert pauc(fpr, tpr, 0.01) == pytest.approx(0.5)            # detection rate 1/2 all the way to 1% false alarms
    assert tpr_at(fpr, tpr, 0.001) == pytest.approx(0.5)
    m = three_d_roc(s, g, auc)
    assert m["auc_d_tau"] == pytest.approx(8 / 9) and m["auc_f_tau"] == pytest.approx(1 / 3)
    assert m["auc_bs"] == pytest.approx(5 / 6 - 1 / 3) and m["snpr"] == pytest.approx(8 / 3)


def test_perfect_random_and_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    g = rng.random(20000) < 0.01
    fpr, tpr = exact_roc(g + rng.random(g.size) * 0.5, g)
    assert auc_exact(fpr, tpr) == 1.0 and pauc(fpr, tpr) == 1.0 and tpr_at(fpr, tpr, 0.001) == 1.0
    s = rng.random(g.size) + g * 0.8
    assert auc_exact(*exact_roc(s, g)) == pytest.approx(roc_auc_score(g, s), abs=1e-9)
    assert pauc(*exact_roc(rng.random(g.size), g)) < 0.05                   # random: about 0.005


def test_chance_by_hand():
    # one pixel dilated by the 4-neighbour cross = 5 px; at the top 1%: 1 - 0.99^5 = 0.04901
    assert chance_hit(5, 1.0) == pytest.approx(1 - 0.99 ** 5)


def test_scene_metrics_objects_groups_and_budgets():
    # Background uniform in [0, 0.5): the 1% cut (100 px) falls inside it, so it is a real threshold. Object 1 scores
    # 5, object 2 0.9 (above every background pixel, inside the top 1% = 100 px), and 10 decoys score 3: the top 0.1%
    # (10 px) is object 1 plus 9 decoys - object 2 is caught at 1% but not at 0.1%.
    rng = np.random.default_rng(0)
    objects = np.zeros((100, 100), np.int32); objects[10, 10] = 1; objects[50, 50] = 2
    score = rng.random((100, 100)) * 0.5; score[10, 10] = 5; score[50, 50] = 0.9; score[np.arange(30, 40), 70] = 3
    metrics = [M(kind="roc_auc"), M(kind="pauc", max_fpr=0.01), M(kind="objects_caught", top_pct=[1.0, 0.1])]
    rows, out = scene_metrics(score, np.ones((100, 100), bool), objects, {1: {"tier": "easy"}, 2: {"tier": "subtle"}},
                              metrics, ["tier"])
    assert [(r["hit_1"], r["hit_0.1"]) for r in rows] == [(True, True), (True, False)]
    assert rows[0]["size_dilated"] == 5 and rows[0]["chance_1"] == pytest.approx(1 - 0.99 ** 5)
    assert out["objects"] == 2 and "pauc_0.01|tier=easy" in out and "pauc_0.01|tier=subtle" in out
    assert out["pauc_0.01|tier=easy"] == pytest.approx(1.0)             # object 1 outranks every background pixel
