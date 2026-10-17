"""detection_measures (app/utils/evaluation/prisma_bench.py): numbers worked by hand."""
import numpy as np
import pytest

from app.utils.evaluation.prisma_bench import detection_measures, score_metrics


def test_by_hand():
    # Targets at 0.9 and 0.7. ROC steps: (0, 1/2), (1/3, 1/2), (1/3, 1), (2/3, 1), (1, 1) -> AUC 1/3*1/2 + 2/3 = 0.8333.
    # Normalised scores 1, .889, .778, .111, 0: targets mean .889, background mean (.889 + .111 + 0) / 3 = .333.
    m = detection_measures(np.array([0.9, 0.8, 0.7, 0.1, 0.0]), np.array([1, 0, 1, 0, 0], bool))
    assert m["auc_exact"] == pytest.approx(5 / 6)
    assert m["pauc_1pct"] == pytest.approx(0.5)          # detection rate 1/2 all the way to 1% false alarms
    assert m["tpr_at_0p1pct"] == pytest.approx(0.5)
    assert m["auc_d_tau"] == pytest.approx(8 / 9) and m["auc_f_tau"] == pytest.approx(1 / 3)
    assert m["auc_bs"] == pytest.approx(5 / 6 - 1 / 3) and m["snpr"] == pytest.approx(8 / 3)


def test_perfect_and_random_and_sklearn():
    rng = np.random.default_rng(0)
    g = rng.random(20000) < 0.01
    perfect = detection_measures(g + rng.random(g.size) * 0.5, g)
    assert perfect["auc_exact"] == 1.0 and perfect["pauc_1pct"] == 1.0 and perfect["tpr_at_0p1pct"] == 1.0
    s = rng.random(g.size) + g * 0.8
    m = detection_measures(s, g)
    from sklearn.metrics import roc_auc_score
    assert m["auc_exact"] == pytest.approx(roc_auc_score(g, s), abs=1e-9)
    assert detection_measures(rng.random(g.size), g)["pauc_1pct"] < 0.05     # random: about 0.005


def test_score_metrics_counts_objects_at_both_budgets():
    # Background uniform in [0, 0.5): the 1% cut (100 px) falls inside it, so it is a real threshold. Object A
    # scores 5, object B 0.9 (above every background pixel, so inside the top 1% = 100 px), and 10 decoys score 3:
    # the top 0.1% (10 px) is A plus 9 decoys - B is caught at 1% but not at 0.1%.
    rng = np.random.default_rng(0)
    gt = np.zeros((100, 100), np.uint8); gt[10, 10] = gt[50, 50] = 1
    score = rng.random((100, 100)) * 0.5; score[10, 10] = 5; score[50, 50] = 0.9; score[np.arange(30, 40), 70] = 3
    m = score_metrics(score, gt, np.ones_like(gt))
    assert (m["objects"], m["objects_caught"], m["objects_caught_0p1"]) == (2, 2, 1)
