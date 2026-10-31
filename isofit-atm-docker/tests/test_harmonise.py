"""harmonise: model file format, band groups, classes, the apply rules, the CLI, and the shipped models."""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from spectral.io import envi

from conftest import ROOT, WL, write_envi
from harmonise import apply, bands, classes, model

SCENE = {"sensor": "ghost", "zmu": np.zeros(4), "zsd": np.ones(4), "zlo": -np.ones(4) * 2, "zhi": np.ones(4) * 2,
         "gate": 1.0}


def _model(path, parts):
    a = {k: np.asarray(v) for k, v in SCENE.items()}
    for name, p in parts.items():
        for k, v in p.items():
            a[f"{name}.{k}"] = np.asarray(v)
    np.savez(path, **a)
    return path


def _own_scale(wl, factor):
    c = np.zeros((len(wl), 2 + 8))
    c[:, 0] = factor
    return {"kind": "own", "wl": wl, "coef": c}


def test_band_groups_partition():
    g = bands.group_masks(WL)
    assert sum(m.sum() for m in g.values()) == len(WL)
    gr = bands.band_groups([450, 900, 940, 1400, 2460, 1000])
    assert list(gr) == ["edge_lo", "core", "weak", "strong", "edge_hi", "core"]


def test_classes():
    from conftest import endmembers
    e = endmembers()
    snow = np.interp(WL, [400, 900, 1300, 1600, 2500], [0.95, 0.85, 0.5, 0.08, 0.05])
    x = np.stack([e["soil"], e["veg"], e["water"], snow])   # (pixels, bands)
    assert list(classes.labels(x, WL)) == ["land", "land", "water", "snow"]


def test_model_load_and_errors(tmp_path):
    p = _model(tmp_path / "m.npz", {"core": _own_scale(WL[:5], 2.0), "edge": {"kind": "identity", "wl": WL[:1]}})
    scene, parts = model.load(p)
    assert set(parts) == {"core", "edge"} and parts["core"]["kind"] == "own" and str(scene["sensor"]) == "ghost"
    bad = _model(tmp_path / "b.npz", {"x": {"kind": "ridge", "wl": WL[:2]}})
    with pytest.raises(ValueError):
        model.load(bad)


def test_apply_own_const_nodata_and_gate():
    x = np.full((4, len(WL)), 0.2)
    parts = {"a": _own_scale(WL[:3], 2.0), "b": {"kind": "const", "wl": WL[3:4], "value": np.array([0.7])},
             "c": {"kind": "nodata", "wl": WL[4:5]}, "d": {"kind": "identity", "wl": WL[5:6]}}
    lab = np.array(["land"] * 4)
    y = apply.harmonise(SCENE, parts, x, WL, [0, 0, 0, 0], lab)
    assert np.allclose(y[:, :3], 0.4) and np.allclose(y[:, 3], 0.7) and np.isnan(y[:, 4]).all()
    assert np.allclose(y[:, 5:], 0.2)
    far = [10, 0, 0, 0]                               # 10 sd out: gated, own skipped, const still applies
    assert apply.gated(SCENE, apply.standardise(SCENE, far)[1])
    y = apply.harmonise(SCENE, parts, x, WL, far, lab)
    assert np.allclose(y[:, :3], 0.2) and np.allclose(y[:, 3], 0.7)


def test_apply_ridge_and_class_parts():
    x = np.full((3, len(WL)), 0.1)
    coef = np.zeros((1 + 4 + 1, 2))
    coef[0] = 1.0          # y = x_in
    coef[-1] = 0.05        # + 0.05
    parts = {"core_water": {"kind": "ridge", "wl": WL[:2], "in_wl": WL[10:11], "coef": coef}}
    lab = np.array(["land", "water", "land"])
    y = apply.harmonise(SCENE, parts, x, WL, [0, 0, 0, 0], lab)
    assert np.allclose(y[1, :2], 0.15) and np.allclose(y[[0, 2], :2], 0.1)
    assert apply.lacks_inputs({"kind": "ridge", "in_wl": np.array([3000.0])}, WL)
    with pytest.raises(ValueError):
        apply.cols(WL, np.array([405.0]))
    with pytest.raises(ValueError):
        apply.fitted({"kind": "const"}, x, WL, np.zeros(4))


def test_cli(tmp_path, rho, monkeypatch):
    m = _model(tmp_path / "m.npz", {"core": _own_scale(WL[:3], 2.0)})
    rfl = str(write_envi(tmp_path / "scn_rfl", rho))
    obs = np.zeros((rho.shape[0], 10, rho.shape[2]), np.float32)
    obs[:, 4] = 30.0
    obsp = str(write_envi(tmp_path / "obs", obs, wl=np.arange(10.0), fwhm=np.ones(10), interleave="bip"))
    rep = tmp_path / "atm_post.json"
    rep.write_text(json.dumps({"sensor": "ghost", "reflectance": rfl, "scene_aot550": 0.2, "scene_h2o": 1.5,
                               "scene_elev_km": 0.1, "aot_prior": None}))
    monkeypatch.setattr(sys, "argv", ["harmonise", str(m), str(rep), obsp])
    runpy.run_module("harmonise", run_name="__main__")
    out = envi.open(rfl + "_harm.hdr", rfl + "_harm").load()
    np.testing.assert_allclose(np.asarray(out)[0, 0, :3], rho[0, :3, 0] * 2, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out)[0, 0, 3:], rho[0, 3:, 0], rtol=1e-6)
    prov = json.loads(Path(rfl + "_harm.json").read_text())
    assert prov["z"][2] == pytest.approx(30 / 90) and not prov["gated"]


@pytest.mark.parametrize("name,sensor", [("enmap_blue600.npz", "enmap"), ("prisma_native.npz", "prisma")])
def test_shipped_models_load(name, sensor):
    scene, parts = model.load(ROOT / "configs" / "harmonise" / name)
    assert str(scene["sensor"]) == sensor and parts
