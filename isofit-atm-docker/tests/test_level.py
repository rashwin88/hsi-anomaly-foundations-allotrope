"""readers.level: the input-level verdict. The one error it must never make is TOA called surface reflectance."""
import numpy as np
import pytest

from conftest import WL, radiance, toa_reflectance
from readers import level


def _pixels(cube_lbs):
    return cube_lbs.transpose(0, 2, 1).reshape(-1, cube_lbs.shape[1])


def test_radiance_is_radiance(rho):
    r = level.classify(_pixels(radiance(rho)), WL)
    assert r["level"] == "radiance"
    assert r["tests"]["shape"]["vote"] == "radiance"


def test_integer_radiance_without_calibration_is_dn(rho):
    r = level.classify(_pixels(np.rint(radiance(rho) * 100)), WL, integer=True)
    assert r["level"] == "dn"


def test_toa_reflectance_is_never_sr(rho):
    r = level.classify(_pixels(toa_reflectance(rho)), WL)
    assert r["level"] == "toa_reflectance"
    assert r["toa_votes"] >= 3
    assert r["tests"]["o2_760"]["vote"] == "toa"


def test_surface_reflectance_passes(rho):
    r = level.classify(_pixels(rho), WL)
    assert r["level"] == "surface_reflectance", r
    assert r["reflectance_scale"] == 1.0
    assert r["sr_tests_run"] >= level.MIN_SR_TESTS


def test_integer_sr_x10000_gets_scale(rho):
    r = level.classify(_pixels(np.rint(rho * 10000)), WL)
    assert r["reflectance_scale"] == pytest.approx(1e-4)
    assert r["level"] == "surface_reflectance"


def test_counts_that_fit_no_scale_are_unknown(rho):
    r = level.classify(_pixels(rho * 1e6), WL)
    assert r["level"] == "unknown"


def test_too_few_tests_means_toa():
    wl = np.r_[np.arange(400.0, 900.0, 10.0), np.arange(2100.0, 2400.0, 10.0)]   # no 940 / 1135 / 1380: O2 only
    from conftest import surface
    r = level.classify(_pixels(surface(wl=wl)), wl)
    assert r["level"] == "toa_reflectance"
    assert "could run" in r["reason"]


def test_few_valid_pixels_is_unknown():
    r = level.classify(np.full((10, len(WL)), np.nan), WL)
    assert r["level"] == "unknown"


def test_depth_needs_bands():
    assert level._depth(np.ones(5), np.arange(5.0), 761, (740, 780)) is None
    spec = np.ones(len(WL))
    spec[np.argmin(abs(WL - 760))] = 0.5
    assert level._depth(spec, WL, 761, (740, 780)) == pytest.approx(0.5)


def test_sample_pixels_masks_fill(rho):
    cube = rho.copy()
    cube[0, :, 0] = -9999
    smp = level.sample_pixels(lambda r0, r1: cube[r0:r1], cube.shape[0], n_blocks=3, rows=2, step=1)
    assert smp.shape[1] == len(WL)
    assert np.isnan(smp).any() and np.nanmin(smp) > -1
