"""
anomaly_scoring's exclude_classes (classes left out at scoring, from an upstream V2 segmentation) and the plain
failure message for a 165-grid band filter under a native-band model.

Reuses the fake-inferencer chain of test_scene_segmentation_v2.py: its class map has rows 0-3 invalid, 4-11 cloud,
12-15 water, 16-17 shadow, the rest land, and thin cloud on row 20. Needs numpy / torch / rasterio, no database.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from test_scene_segmentation_v2 import (  # noqa: F401  (chain is a fixture)
    BF_ID, SEG_ID, _class_map, _read, _score_cfg, chain,
)
from allotrope.action_types import anomaly_scoring, scene_segmentation  # noqa: E402

SCORE = ("models", "indradhanu_v2", "anomaly_score.tif")


def _segment(chain, **cfg):
    seg_ctx = chain.ctx({"input_band_filter_output_id": BF_ID, "method": "indradhanu_v2",
                         "exclude_classes": ["cloud"], "thin_cloud_is_cloud": False,
                         "model_codename": "Indradhanu V2", **cfg}, "seg_out")
    scene_segmentation.run(seg_ctx)
    chain.dirs[SEG_ID] = seg_ctx.output_dir
    return seg_ctx


def _score(chain, name, exclude=None):
    from allotrope.action_types import _anomaly_scoring_run as scoring

    cfg = _score_cfg(seg=True)
    if exclude is not None:
        cfg["exclude_classes"] = exclude
    ctx = chain.ctx(cfg, name)
    scoring.run(ctx)
    return ctx.output_dir


def test_keep_mask_from_classes_reads_class_codes(chain):
    from allotrope.action_types._anomaly_scoring_run import keep_mask_for_classes

    seg = _segment(chain).output_dir
    cm = _class_map()
    valid = cm != 255
    np.testing.assert_array_equal(keep_mask_for_classes(seg, ["cloud"]), (valid & (cm != 2)).astype(np.uint8))
    np.testing.assert_array_equal(keep_mask_for_classes(seg, ["cloud", "water"]),
                                  (valid & ~np.isin(cm, [1, 2])).astype(np.uint8))
    thin = keep_mask_for_classes(seg, ["thin_cloud"])
    assert thin[20].sum() == 0 and thin[21].all() and thin[4:12].all()     # haze row out, thick cloud kept
    np.testing.assert_array_equal(keep_mask_for_classes(seg, []), valid.astype(np.uint8))


def test_none_keeps_todays_behaviour(chain):
    _segment(chain)
    before = _score(chain, "s_none")
    explicit = _score(chain, "s_cloud", exclude=["cloud"])     # the segmentation's own choice, spelled out
    np.testing.assert_array_equal(_read(before.joinpath(*SCORE)), _read(explicit.joinpath(*SCORE)))
    s = json.loads((before / "summary.json").read_text())
    assert s["exclude_classes"] is None and "keep_mask" in s["keep_source"]
    stats = json.loads(before.joinpath("models", "indradhanu_v2", "stats.json").read_text())
    assert stats["exclude_classes_at_scoring"] is None and stats["excluded_classes"] == ["cloud"]


def test_leaving_water_out_zeroes_it_and_records_the_choice(chain):
    _segment(chain)
    a = _read(_score(chain, "s_a", exclude=["cloud"]).joinpath(*SCORE))
    out_b = _score(chain, "s_b", exclude=["cloud", "water"])
    b = _read(out_b.joinpath(*SCORE))
    assert (b[12:16] == 0).all() and a[12:16].any()          # water scored in (a), zero in (b)
    np.testing.assert_allclose(b[18:], a[18:], rtol=1e-6)     # V2 scores the same, then zeroes
    stats = json.loads(out_b.joinpath("models", "indradhanu_v2", "stats.json").read_text())
    assert stats["excluded_classes"] == ["cloud", "water"] and stats["exclude_classes_at_scoring"] == ["cloud", "water"]
    summary = json.loads((out_b / "summary.json").read_text())
    assert summary["exclude_classes"] == ["cloud", "water"]
    assert summary["kept_pct"] == pytest.approx(100 * (32 - 4 - 8 - 4) / 28, abs=0.01)


def test_exclude_needs_a_v2_segmentation(chain, tmp_path):
    from allotrope.action_types._anomaly_scoring_run import run
    from allotrope.action_types._meta import PlainActionError

    index_seg = tmp_path / "index_seg"
    index_seg.mkdir()
    (index_seg / "summary.json").write_text(json.dumps({"method": "indices"}))
    chain.dirs[SEG_ID] = index_seg
    ctx = chain.ctx({**_score_cfg(seg=True), "exclude_classes": ["water"]}, "s_idx")
    with pytest.raises(PlainActionError, match="Indradhanu V2"):
        run(ctx)


def test_validate_config_checks_the_classes(monkeypatch):
    from allotrope.foundation_models import resolver
    from test_scene_segmentation_v2 import _model

    monkeypatch.setattr(resolver, "list_catalog", lambda root: [_model()])
    base = {"input_band_filter_output_id": BF_ID, "model_codenames": ["Indradhanu V2"]}
    assert anomaly_scoring.validate_config(base, "prisma")["exclude_classes"] is None
    ok = anomaly_scoring.validate_config({**base, "input_scene_segmentation_output_id": SEG_ID,
                                          "exclude_classes": ["cloud", "thin_cloud"]}, "prisma")
    assert ok["exclude_classes"] == ["cloud", "thin_cloud"]
    with pytest.raises(ValueError, match="unknown classes"):
        anomaly_scoring.validate_config({**base, "input_scene_segmentation_output_id": SEG_ID,
                                         "exclude_classes": ["land"]}, "prisma")
    with pytest.raises(ValueError, match="scene segmentation input"):
        anomaly_scoring.validate_config({**base, "exclude_classes": ["water"]}, "prisma")


# --- addition C: the 165-grid band filter under V2 ---------------------------


def test_old_grid_failure_is_plain():
    from allotrope.action_types._anomaly_scoring_run import OLD_GRID_MESSAGE, _check_band_grid
    from allotrope.action_types._meta import PlainActionError
    from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID

    with pytest.raises(PlainActionError) as exc:
        _check_band_grid("Indradhanu V2", "native", np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, float))
    first = str(exc.value).splitlines()[0]
    assert first == OLD_GRID_MESSAGE
    assert "old 165-band grid" in first and "Band filter (native bands)" in first
