"""
scene_segmentation's "indradhanu_v2" method and anomaly_scoring's use of it.

The model is replaced by a fake inferencer with a fixed class map; what is tested is the product
contract: old configs stay "indices", keep_mask from the class map, the summary's percentages, and
that anomaly_scoring takes the upstream classes instead of running V2's head again - with scores
equal to the chain where it did run its own head. Needs numpy / torch / rasterio / PIL, no database.
"""
from __future__ import annotations

import json
import os
import pickle
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# allotrope.config builds Settings at import and requires these; nothing here connects.
os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")

from allotrope.action_types import scene_segmentation  # noqa: E402
from allotrope.action_types._scene_segmentation_run import (  # noqa: E402
    INVALID_CODE, class_summary, keep_mask_from_class_map,
)

C, H, W = 12, 32, 32
BF_ID = "output_" + str(uuid.uuid4())
SEG_ID = "output_" + str(uuid.uuid4())


def _class_map():
    """Rows 0-3 invalid (255), 4-11 cloud, 12-15 water, 16-17 shadow, rest land."""
    cm = np.zeros((H, W), np.uint8)
    cm[4:12] = 2
    cm[12:16] = 1
    cm[16:18] = 3
    cm[:4] = INVALID_CODE
    return cm


# --- config -------------------------------------------------------------


def test_config_without_method_is_indices():
    out = scene_segmentation.validate_config({"input_band_filter_output_id": BF_ID}, "prisma")
    assert out["method"] == "indices"
    assert out["classes_to_mask"] == ["water", "cloud", "shadow", "vegetation"]
    assert out["thresholds"]["ndwi_water"] == 0.3


def test_hyperspectral_templates_run_v2():
    for sensor in ("prisma", "enmap", "aviris_ng", "tanager"):
        d = scene_segmentation.META.default_config_per_sensor[sensor]
        assert d == {"method": "indradhanu_v2", "exclude_classes": ["cloud"],
                     "thin_cloud_is_cloud": False, "model_codename": "Indradhanu V2"}


def test_v2_config_checks_classes_and_model(monkeypatch):
    from allotrope.foundation_models import resolver

    monkeypatch.setattr(resolver, "list_catalog", lambda root: [_model()])
    base = {"input_band_filter_output_id": BF_ID, "method": "indradhanu_v2"}
    out = scene_segmentation.validate_config(base, "prisma")
    assert (out["exclude_classes"], out["thin_cloud_is_cloud"], out["model_codename"]) == (
        ["cloud"], False, "Indradhanu V2")
    with pytest.raises(ValueError, match="exclude_classes"):
        scene_segmentation.validate_config({**base, "exclude_classes": ["vegetation"]}, "prisma")
    with pytest.raises(ValueError, match="no segmentation head"):
        scene_segmentation.validate_config({**base, "model_codename": "Indradhanu"}, "prisma")


# --- keep_mask and summary ------------------------------------------------


def test_keep_mask_excludes_only_the_chosen_classes():
    cm = _class_map()
    keep = keep_mask_from_class_map(cm, ["cloud"])
    assert keep.dtype == np.uint8
    assert not keep[:12].any()                  # invalid + cloud
    assert keep[12:].all()                      # water, shadow, land kept
    keep2 = keep_mask_from_class_map(cm, ["cloud", "water"])
    assert not keep2[12:16].any() and keep2[16:].all()
    assert keep_mask_from_class_map(cm, []).sum() == (H - 4) * W


def test_summary_percentages():
    cm = _class_map()
    haze = np.zeros((H, W), bool)
    haze[20:21] = True                          # one row of land under haze
    s = class_summary(cm, ["cloud"], haze)
    n_valid = (H - 4) * W
    assert s["pixel_counts"]["spatial_valid"] == n_valid
    assert s["class_pct_of_valid"]["cloud"] == round(100 * 8 / 28, 3)
    assert s["class_pct_of_valid"]["water"] == round(100 * 4 / 28, 3)
    assert s["class_pct_of_valid"]["land"] == round(100 * 14 / 28, 3)
    assert s["class_pct_of_valid"]["thin_cloud"] == round(100 * 1 / 28, 3)
    assert s["excluded_pct_of_valid"] == round(100 * 8 / 28, 3)
    assert s["kept_pct_of_valid"] == round(100 * 20 / 28, 3)
    excluded = {e["key"] for e in s["legend"] if e["excluded"]}
    assert excluded == {"cloud"}
    assert sum(s["class_pct_of_valid"][k] for k in ("land", "water", "cloud", "shadow", "snow")) == pytest.approx(100, abs=0.01)


# --- run: segmentation, then scoring with and without it -------------------


def _model():
    from allotrope.foundation_models.resolver import ResolvedModel

    return ResolvedModel(
        architecture="spectral_stem_finetune", codename="Indradhanu V2", codename_script="", sensor="hyperspectral",
        label="", foundation_model_name="spectral_stem_finetune", checkpoint_abs_path="x.pt",
        checkpoint_filename="x.pt", model_config={}, scoring_methods=("residual_mahalanobis",),
        default_scoring_method="residual_mahalanobis", default_patch_size=128, default_stride=32,
        default_batch_size=4, valid_patch_sizes=(128,), pixel_stats_abs_path=None, band_grid="native",
        exclude_classes=("cloud",), thin_cloud_is_cloud=False)


class _FakeInferencer:
    CLASS_NAMES = ("land", "water", "cloud", "shadow", "snow")
    SEG_CLASSES = ("water", "cloud", "thin_cloud", "shadow", "snow")
    segment_calls = 0

    def __init__(self, config):
        import torch
        self.device = torch.device("cpu")

    def read_bands(self, cube, wl):
        return cube, np.asarray(wl), np.arange(len(wl))

    def predict_full_scene(self, scene, mask):
        g = np.random.default_rng(1)
        return scene * 0.9 + 0.01 * __import__("torch").from_numpy(g.normal(size=tuple(scene.shape)).astype(np.float32))

    def segment_full_scene(self, cube, valid, thin_cloud=True):
        type(self).segment_calls += 1
        cm = _class_map().astype(np.int64)
        cm[cm == INVALID_CODE] = 0
        prob = np.zeros((5, H, W), np.float32)
        prob[2, 20] = 0.9                       # thin cloud over one land row
        return cm * valid, prob


@pytest.fixture
def chain(tmp_path, monkeypatch):
    import app.foundation_models.inferencers.inferencer_factory as factory
    import app.models.training.inference_config as ic
    from allotrope.foundation_models import resolver

    monkeypatch.setattr(resolver, "list_catalog", lambda root: [_model()])
    monkeypatch.setattr(factory, "get_inferencer", lambda cfg: _FakeInferencer(cfg))
    monkeypatch.setattr(ic, "InferenceConfig", lambda **kw: SimpleNamespace(**kw))
    # The score / recon PNG renderers use matplotlib.cm.get_cmap, gone from the matplotlib a dev venv may
    # have (the worker image pins an older one); they are decoration, not the contract tested here.
    from allotrope.action_types import _anomaly_scoring_run as scoring
    for name in ("_render_score_png", "_render_recon_png", "_render_thumbnail"):
        monkeypatch.setattr(scoring, name, lambda *a, **k: None)
    _FakeInferencer.segment_calls = 0

    g = np.random.default_rng(0)
    cube = (0.2 + 0.05 * g.random((C, H, W))).astype(np.float32)
    validity = np.ones((C, H, W), np.int8)
    validity[:, :4] = 0
    vendable = SimpleNamespace(normalized_hyperspectral_cube=cube, validity_cube=validity,
                               band_cw_order=list(np.linspace(450, 2400, C)), units="reflectance")
    dirs = {BF_ID: tmp_path / "bf", SEG_ID: tmp_path / "seg"}
    for d in dirs.values():
        d.mkdir()
    with (dirs[BF_ID] / "filtered_vendable.pkl").open("wb") as f:
        pickle.dump(vendable, f)

    def ctx(cfg, out):
        out = tmp_path / out
        out.mkdir()
        return SimpleNamespace(configuration=cfg, sensor_type="prisma", scene_id=uuid.uuid4(),
                               action_id=uuid.uuid4(), data_dir=tmp_path, output_dir=out,
                               on_step=lambda s: None, resolve_action_output=lambda oid: dirs[oid])
    return SimpleNamespace(ctx=ctx, dirs=dirs, tmp=tmp_path)


def _score_cfg(seg: bool):
    cfg = {"input_band_filter_output_id": BF_ID, "model_codenames": ["Indradhanu V2"], "model_overrides": {}}
    if seg:
        cfg["input_scene_segmentation_output_id"] = SEG_ID
    return cfg


def _read(path: Path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(1)


def test_v2_segmentation_writes_its_outputs(chain):
    from allotrope.action_types import _scene_segmentation_run as v2

    seg_ctx = chain.ctx({"input_band_filter_output_id": BF_ID, "method": "indradhanu_v2",
                         "exclude_classes": ["cloud"], "thin_cloud_is_cloud": False,
                         "model_codename": "Indradhanu V2"}, "seg_out")
    scene_segmentation.run(seg_ctx)
    out = seg_ctx.output_dir
    for f in ("class_map.tif", "keep_mask.tif", "class_map.png", "class_map_overlay.png", "rgb.png",
              "preview.png", "class_codes.bin", "summary.json"):
        assert (out / f).exists(), f
    np.testing.assert_array_equal(_read(out / "class_map.tif"), _class_map())
    np.testing.assert_array_equal(_read(out / "keep_mask.tif"), keep_mask_from_class_map(_class_map(), ["cloud"]))
    codes = np.fromfile(out / "class_codes.bin", np.uint8).reshape(H, W)
    assert codes[20, 0] == v2.HAZE_BIT and codes[21, 0] == 0 and codes[0, 0] == INVALID_CODE
    s = json.loads((out / "summary.json").read_text())
    assert s["method"] == "indradhanu_v2" and s["scene_shape"] == [H, W]
    assert s["excluded_pct_of_valid"] == round(100 * 8 / 28, 3)
    assert scene_segmentation.summarize(seg_ctx, out)["action_id"] == f"action_{seg_ctx.action_id}"


def test_scoring_uses_upstream_v2_segmentation_instead_of_its_head(chain):
    from allotrope.action_types import _anomaly_scoring_run as scoring

    seg_ctx = chain.ctx({"input_band_filter_output_id": BF_ID, "method": "indradhanu_v2",
                         "exclude_classes": ["cloud"], "thin_cloud_is_cloud": False,
                         "model_codename": "Indradhanu V2"}, "seg_out")
    scene_segmentation.run(seg_ctx)
    chain.dirs[SEG_ID] = seg_ctx.output_dir
    assert _FakeInferencer.segment_calls == 1

    # Before: no upstream segmentation, V2 runs its own head.
    alone = chain.ctx(_score_cfg(seg=False), "score_alone")
    scoring.run(alone)
    assert _FakeInferencer.segment_calls == 2
    # After: the upstream V2 segmentation, and the head does not run again.
    withseg = chain.ctx(_score_cfg(seg=True), "score_with_seg")
    scoring.run(withseg)
    assert _FakeInferencer.segment_calls == 2

    m_alone = alone.output_dir / "models" / "indradhanu_v2"
    m_seg = withseg.output_dir / "models" / "indradhanu_v2"
    stats = json.loads((m_seg / "stats.json").read_text())
    assert stats["segmentation"] == f"from upstream scene_segmentation action_{seg_ctx.action_id}"
    assert stats["excluded_classes"] == ["cloud"] and stats["cloud_rule"] == "thick"
    assert json.loads((m_alone / "stats.json").read_text())["segmentation"] == "internal head"
    s_alone, s_seg = _read(m_alone / "anomaly_score.tif"), _read(m_seg / "anomaly_score.tif")
    np.testing.assert_allclose(s_seg, s_alone, rtol=1e-6, atol=1e-7)
    assert (s_seg[4:12] == 0).all() and s_seg[12:].any()
    np.testing.assert_array_equal(_read(m_seg / "class_map.tif"), _read(m_alone / "class_map.tif"))
    np.testing.assert_array_equal(_read(m_seg / "mask_cloud.tif"), _read(m_alone / "mask_cloud.tif"))
    summary = json.loads((withseg.output_dir / "summary.json").read_text())
    assert summary["upstream_segmentation"]["method"] == "indradhanu_v2"
