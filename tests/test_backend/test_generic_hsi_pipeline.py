"""
generic_hsi through the real action code: band_filter_apply.run -> anomaly_scoring run (MNF-RX), in-process.

The unit suite (tests/test_utils/test_generic_hsi.py) proves the reader; this
proves the wiring the sensor checklist touches (source_path, the band-filter
builder branch, the defaults, the not-reflectance rule in scoring, scene
onboarding's id and the export georef) on a few synthetic surprise files.
Foundation models are not run: the catalog holds only MNF-RX and a V2
manifest without weights, which the not-reflectance rule must skip before
loading anything.
"""

import json
import os
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("POSTGRES_PASSWORD", "unused")
os.environ.setdefault("JWT_SECRET", "unused")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test_utils"))

from generic_hsi_files import GEO_TRANSFORM, UTM_TRANSFORM, WL, H, W, cube_for, swath, wl_tags, write_envi, write_gtiff, write_igm  # noqa: E402

from allotrope.action_types import anomaly_scoring, band_filter_apply  # noqa: E402
from allotrope.action_types._anomaly_scoring_run import run as scoring_run  # noqa: E402
from allotrope.config import settings  # noqa: E402
from allotrope.sensors.source_path import resolve_source_path  # noqa: E402
from allotrope_worker.action_run import ActionRunContext  # noqa: E402
from allotrope_worker.scene_onboard import _filename_for_id, _native_projection, _build  # noqa: E402

MNF_RX = {"architecture": "mnf_rx", "label": "MNF + Global RX", "sensor": "hyperspectral", "family": "classical",
          "codename": {"name": "MNF-RX", "script": "", "meaning": "x", "why": "x"},
          "current": {"file": "", "version": "v1"}}
V2 = {"architecture": "spectral_stem_finetune", "label": "x", "sensor": "hyperspectral",
      "codename": {"name": "Indradhanu V2", "script": "x", "meaning": "x", "why": "x"},
      "current": {"file": "missing.pt", "model_config": {"model_type": "spectral_stem_finetune",
                                                          "stem": "reduced_attention", "seg_classes": 5}}}


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    root = tmp_path / "models"
    for manifest in (MNF_RX, V2):
        (root / manifest["architecture"]).mkdir(parents=True)
        (root / manifest["architecture"] / "current.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(settings, "models_dir", str(root))
    import matplotlib
    from matplotlib import cm

    if not hasattr(cm, "get_cmap"):
        # The score renderer calls cm.get_cmap, removed in matplotlib 3.9; the worker image pins an older one.
        monkeypatch.setattr(cm, "get_cmap", lambda name, lut=None: matplotlib.colormaps[name].resampled(lut or 256),
                            raising=False)
    return root


def ctx(tmp_path, raw, cfg, outputs=None, name="out"):
    out = tmp_path / name
    out.mkdir()
    return ActionRunContext(
        action_id=uuid.uuid4(), project_id=uuid.uuid4(), scene_id=uuid.uuid4(), sensor_type="generic_hsi",
        configuration=cfg, data_dir=tmp_path, artifacts_dir=tmp_path, output_dir=out, output_dir_rel=name,
        scene_raw_path=raw, resolve_action_output=lambda wire: outputs[wire],
    )


def anomalous_cube(scale=1.0, dtype="float32"):
    cube = cube_for(WL).astype(np.float64)
    rng = np.random.default_rng(0)
    cube += rng.normal(0, 0.002, cube.shape)
    cube[:, 5, 7] = np.linspace(0.6, 0.1, WL.size)      # one odd spectrum
    return (cube * scale).astype(dtype)


def band_filter_then_mnf_rx(tmp_path, raw, codenames):
    cfg_bf = band_filter_apply.validate_config({**band_filter_apply.META.default_config_per_sensor["generic_hsi"],
                                                "input_scene_id": "scene_" + str(uuid.uuid4())},
                                               "generic_hsi")
    bf_ctx = ctx(tmp_path, raw, cfg_bf, name="bf")
    band_filter_apply.run(bf_ctx)
    summary = json.loads((bf_ctx.output_dir / "summary.json").read_text())
    cfg = {"input_band_filter_output_id": "output_" + str(uuid.uuid4()), "model_codenames": codenames,
           "model_overrides": {}}
    sc_ctx = ctx(tmp_path, raw, cfg, outputs={cfg["input_band_filter_output_id"]: bf_ctx.output_dir}, name="sc")
    scoring_run(sc_ctx)
    return summary, json.loads((sc_ctx.output_dir / "summary.json").read_text()), sc_ctx.output_dir


@pytest.mark.parametrize("case", ["ghost_cog_utm", "osk_envi_igm"])
def test_reflectance_scene_band_filter_then_mnf_rx(tmp_path, models_dir, case):
    raw = tmp_path / "raw"
    raw.mkdir()
    if case == "ghost_cog_utm":
        write_gtiff(raw / "dataset1.tif", anomalous_cube(), band_tags=wl_tags(WL), nodata=0,
                    crs="EPSG:32613", transform=UTM_TRANSFORM, cog=True)
    else:
        write_envi(raw, "osk", anomalous_cube(), interleave="bsq", wl=WL, fwhm=[8.7] * WL.size)
        write_igm(raw, "osk", *swath())
    src = resolve_source_path("generic_hsi", raw)
    bf, scoring, out = band_filter_then_mnf_rx(tmp_path, raw, ["MNF-RX"])
    assert bf["sensor"] == "generic_hsi" and bf["scene_units"] is None and bf["band_count"] < WL.size
    assert scoring["models_skipped_not_reflectance"] == []
    import rasterio

    with rasterio.open(out / "models" / "MNF-RX" / "anomaly_score.tif") as s:
        score = s.read(1)
    assert np.unravel_index(np.nanargmax(score), score.shape) == (5, 7)
    from app.georef.from_scene_file import resolve_scene_georef

    _, crs = resolve_scene_georef(scene_dir=raw, sensor_type="generic_hsi", target_shape=(H, W))
    assert crs.to_epsg() == (32613 if case == "ghost_cog_utm" else 4326)
    assert _filename_for_id("generic_hsi", [p.relative_to(raw) for p in raw.iterdir()], root=raw) == src.stem
    assert _native_projection("generic_hsi", _build("generic_hsi", src)) == f"EPSG:{crs.to_epsg()}"


def test_not_reflectance_scene_scores_mnf_rx_only(tmp_path, models_dir):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gtiff(raw / "dn.tif", anomalous_cube(scale=150000, dtype="float32") + 20000, band_tags=wl_tags(WL))
    default = anomaly_scoring.META.default_config_per_sensor["generic_hsi"]["model_codenames"]
    assert default == ["Indradhanu V2", "MNF-RX"]
    bf, scoring, out = band_filter_then_mnf_rx(tmp_path, raw, default)
    assert bf["scene_units"] == "not_reflectance"
    assert scoring["scene_units"] == "not_reflectance"
    assert scoring["models_skipped_not_reflectance"] == ["Indradhanu V2"]
    assert scoring["model_codenames"] == ["MNF-RX"]          # what prep reads; a skipped model has no raster
    assert (out / "models" / "MNF-RX" / "anomaly_score.tif").is_file()
    from allotrope.action_types import anomaly_detection_prep

    sc_id = "output_" + str(uuid.uuid4())
    cfg = anomaly_detection_prep.validate_config(
        {**anomaly_detection_prep.META.default_config_per_sensor["generic_hsi"],
         "input_anomaly_scoring_output_id": sc_id}, "generic_hsi")
    prep_ctx = ctx(tmp_path, raw, cfg, outputs={sc_id: out}, name="prep")
    anomaly_detection_prep.run(prep_ctx)
    assert (prep_ctx.output_dir / "composite_score.tif").is_file()


def test_not_reflectance_with_foundation_models_only_refuses(tmp_path, models_dir):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gtiff(raw / "x.tif", anomalous_cube(), band_tags=wl_tags(WL))
    (raw / "allotrope_ingest.json").write_text(json.dumps({"scale": "radiance"}))
    with pytest.raises(ValueError, match="not reflectance"):
        band_filter_then_mnf_rx(tmp_path, raw, ["Indradhanu V2"])


def test_onboarding_vendable_pickles_with_units(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gtiff(raw / "x.tif", anomalous_cube(), band_tags=wl_tags(WL), crs="EPSG:4326", transform=GEO_TRANSFORM)
    builder = _build("generic_hsi", resolve_source_path("generic_hsi", raw))
    v = pickle.loads(pickle.dumps(builder.vend_dataset()))
    assert v.units is None and builder.preflight["georef"]["exportable"]


def _not_reflectance_band_filter(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_gtiff(raw / "x.tif", anomalous_cube(), band_tags=wl_tags(WL))
    (raw / "allotrope_ingest.json").write_text(json.dumps({"scale": "radiance"}))
    cfg = band_filter_apply.validate_config({**band_filter_apply.META.default_config_per_sensor["generic_hsi"],
                                             "input_scene_id": "scene_" + str(uuid.uuid4())}, "generic_hsi")
    bf_ctx = ctx(tmp_path, raw, cfg, name="bf")
    band_filter_apply.run(bf_ctx)
    return raw, bf_ctx.output_dir


def _segment(tmp_path, raw, bf_dir, overrides):
    from allotrope.action_types import scene_segmentation

    bf_id = "output_" + str(uuid.uuid4())
    cfg = scene_segmentation.validate_config(
        {**scene_segmentation.META.default_config_per_sensor["generic_hsi"], **overrides,
         "input_band_filter_output_id": bf_id}, "generic_hsi")
    seg_ctx = ctx(tmp_path, raw, cfg, outputs={bf_id: bf_dir}, name="seg")
    scene_segmentation.run(seg_ctx)
    return seg_ctx.output_dir


def test_v2_segmentation_refuses_a_not_reflectance_scene(tmp_path, models_dir):
    raw, bf_dir = _not_reflectance_band_filter(tmp_path)
    with pytest.raises(ValueError, match="V2 segmentation needs reflectance.*use method 'indices'"):
        _segment(tmp_path, raw, bf_dir, {"method_if_not_reflectance": None})


def test_generic_hsi_segmentation_default_falls_back_to_indices(tmp_path, models_dir):
    from allotrope.action_types import scene_segmentation

    default = scene_segmentation.META.default_config_per_sensor["generic_hsi"]
    assert (default["method"], default["method_if_not_reflectance"]) == ("indradhanu_v2", "indices")
    raw, bf_dir = _not_reflectance_band_filter(tmp_path)
    out = _segment(tmp_path, raw, bf_dir, {})
    assert (out / "keep_mask.tif").is_file()
