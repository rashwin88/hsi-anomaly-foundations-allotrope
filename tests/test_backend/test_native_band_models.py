"""
Native-band models in the product (Indradhanu V2 = Indradhanu Heavy since 2026-10-29, FT04 before;
docs/proposals/ship-ft04.md): the resolver reads the embedded model_config from the manifest, anomaly_scoring refuses
a band filter on the wrong grid, and the excluded cloud is thick cloud only.
"""

import json

import numpy as np
import pytest

from allotrope.action_types._anomaly_scoring_run import _check_band_grid
from allotrope.foundation_models.resolver import _from_manifest, _model_config_from_manifest
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID

GRID = np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID, dtype=np.float64)
NATIVE = np.linspace(2450.0, 400.0, 230)      # PRISMA-like: native and descending
CONFIG = {"model_type": "spectral_stem_finetune", "stem": "reduced_attention", "seg_classes": 5}


def manifest(tmp_path, current):
    path = tmp_path / "current.json"
    path.write_text(json.dumps({
        "architecture": "spectral_stem_finetune", "label": "x", "sensor": "hyperspectral",
        "codename": {"name": "Indradhanu V2", "script": "x", "meaning": "x", "why": "x"}, "current": current}))
    return path


def test_resolver_reads_embedded_config_and_capabilities(tmp_path):
    m = _from_manifest(tmp_path, manifest(tmp_path, {"file": "m.pt", "model_config": CONFIG}))
    assert m.model_config == CONFIG
    assert (m.band_grid, m.exclude_classes, m.foundation_model_name) == ("native", ("cloud",), "spectral_stem_finetune")
    assert m.thin_cloud_is_cloud is False


def test_thick_cloud_rule_ignores_thin_cloud():
    """segment_full_scene(thin_cloud=False), what anomaly_scoring passes for Indradhanu V2, hands a pixel that is only
    thin cloud to the next class by priority and keeps thick cloud as cloud. A stub head echoes its input as
    probabilities, so the cube's 5 channels are the head's outputs (water, cloud, thin_cloud, shadow, snow)."""
    import torch
    from types import SimpleNamespace
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer

    net = SimpleNamespace(seg_head=True, segment=lambda x, wl: torch.logit(x.clamp(1e-4, 1 - 1e-4)))
    net.eval = lambda: net
    inf = SpectralStemInferencer.__new__(SpectralStemInferencer)
    inf.model, inf.device = SimpleNamespace(inner=net, wavelengths_nm=None), torch.device("cpu")
    # pixels: thick cloud, thin cloud over water, thin cloud alone
    cube = np.array([[0, .9, 0], [.9, 0, 0], [0, .9, .9], [0, 0, 0], [0, 0, 0]], np.float32).reshape(5, 1, 3)
    valid = np.ones((1, 3), bool)
    assert inf.segment_full_scene(cube, valid)[0].tolist() == [[2, 2, 2]]
    assert inf.segment_full_scene(cube, valid, thin_cloud=False)[0].tolist() == [[2, 1, 0]]


@pytest.mark.parametrize("current", [{}, {"model_config": {"model_type": "hyperspectral_segformer_mae"}}])
def test_resolver_refuses_missing_or_mismatched_config(current):
    with pytest.raises(KeyError):
        _model_config_from_manifest("spectral_stem_finetune", current)


@pytest.mark.parametrize("band_grid,wavelengths", [("native", NATIVE), ("common165", GRID)])
def test_band_grid_accepts_its_own_grid(band_grid, wavelengths):
    _check_band_grid("m", band_grid, wavelengths)


@pytest.mark.parametrize("band_grid,wavelengths", [("native", GRID), ("common165", NATIVE)])
def test_band_grid_refuses_the_other_grid(band_grid, wavelengths):
    with pytest.raises(ValueError, match="use_common_wavelength_grid"):
        _check_band_grid("m", band_grid, wavelengths)
