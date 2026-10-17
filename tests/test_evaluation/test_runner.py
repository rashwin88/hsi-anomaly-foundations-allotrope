"""End to end: a synthetic scene, a stand-in model, every scorer kind, caching and the report."""
import json

import numpy as np
import pytest

from app.evaluation import Evaluation, EvaluationConfig
from app.evaluation.datasets import EvalScene


class FakeModel:
    """Reconstructs the background exactly (a constant 0.2 cube) and calls the left third of the scene water."""
    identity = "fake-1"

    def __init__(self):
        self.calls = 0

    def reconstruct(self, scene):
        self.calls += 1
        return scene.cube, np.full_like(scene.cube, 0.2), scene.valid

    def segment(self, scene):
        cls = np.zeros(scene.valid.shape, np.int64)
        cls[:, : scene.valid.shape[1] // 3] = 1
        return cls


def scene():
    rng = np.random.default_rng(0)
    C, H, W = 8, 90, 90
    cube = (0.2 + 0.01 * rng.standard_normal((C, H, W))).astype(np.float32)
    objects = np.zeros((H, W), np.int32)
    for i, (y, x, amp) in enumerate([(20, 50, 0.3), (60, 70, 0.08), (45, 10, 0.3)], 1):   # object 3 sits on "water"
        objects[y, x] = i
        cube[:, y, x] += amp
    attrs = {1: {"tier": "easy"}, 2: {"tier": "subtle"}, 3: {"tier": "easy"}}
    return EvalScene("synthetic", "s1", "test", cube, np.linspace(450, 2400, C), np.ones((H, W), bool), objects, attrs)


def config(tmp_path):
    return EvaluationConfig(**{
        "name": "e2e", "models": [{"name": "fake", "checkpoint": "unused"}],
        "datasets": [{"kind": "implant", "path": "unused"}],
        "scorers": [{"kind": "residual_mahalanobis", "d": 4, "local_window": None},
                    {"kind": "L1"},
                    {"kind": "pca_rx", "d": 4, "local_window": None},
                    {"kind": "exclude", "classes": ["water"], "segmenter": "fake",
                     "base": {"kind": "residual_mahalanobis", "d": 4, "local_window": None}}],
        "group_by": ["tier"], "output": str(tmp_path / "out")})


def test_end_to_end_and_cache(tmp_path):
    cfg, model = config(tmp_path), FakeModel()
    summary = Evaluation(cfg, datasets=[[scene()]], models={"fake": model}).run()
    out = tmp_path / "out"
    for f in ("per_object.csv", "per_scene.csv", "summary.json", "report.md", "config.json", "provenance.json"):
        assert (out / f).exists(), f
    res = summary["synthetic"]
    assert set(res) == {"fake | residual_mahalanobis(d=4,w=global)", "fake | L1", "- | pca_rx(d=4,w=global)",
                        "fake | exclude(water;fit)[residual_mahalanobis(d=4,w=global)]"}
    rm = res["fake | residual_mahalanobis(d=4,w=global)"]
    assert rm["objects"] == 3 and rm["caught"]["1"] == 1.0          # all three are 0.08-0.3 above a 0.01 noise floor
    ex = res["fake | exclude(water;fit)[residual_mahalanobis(d=4,w=global)]"]
    assert ex["caught"]["1"] == pytest.approx(2 / 3)                # object 3 sits on "water": scored 0, missed
    assert ex["groups"]["tier=easy"]["caught"]["1"] == pytest.approx(0.5)
    assert "tier=easy" in rm["groups"] and "tier=subtle" in rm["groups"]
    assert model.calls == 1                                          # one reconstruction per scene and model
    Evaluation(cfg, datasets=[[scene()]], models={"fake": model}).run()
    assert model.calls == 1                                          # second run: every score map from the cache
    assert "| *chance* |" in (out / "report.md").read_text()


def test_config_validation(tmp_path):
    base = {"name": "x", "datasets": [{"kind": "implant", "path": "p"}], "output": str(tmp_path)}
    with pytest.raises(ValueError):                                  # exclude without segmenter
        EvaluationConfig(**base, models=[{"name": "m", "checkpoint": "c"}],
                         scorers=[{"kind": "exclude", "classes": ["water"], "base": {"kind": "L1"}}])
    with pytest.raises(ValueError):                                  # segmenter not a model
        EvaluationConfig(**base, models=[{"name": "m", "checkpoint": "c"}],
                         scorers=[{"kind": "class_aware", "segmenter": "other"}])
    assert EvaluationConfig(**base, scorers=[{"kind": "pca_rx"}]).models == []   # classical only: no model needed
