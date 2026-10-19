"""
Native-band models in the product (Indradhanu V2 = FT04; docs/proposals/ship-ft04.md): the resolver reads the
embedded model_config from the manifest, and anomaly_scoring refuses a band filter on the wrong grid.
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
