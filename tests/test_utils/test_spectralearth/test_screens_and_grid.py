"""SpectralEarth screens and the bridge onto the 165-band common grid."""

from dataclasses import dataclass

import numpy as np
import pytest

from app.models.dataset.spectralearth import SpectralEarthPatch, load_band_table
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID
from app.utils.spectralearth.screens import NodataScreen, ScreenChain

NODATA = -32768


def _reflectance(wl: np.ndarray) -> np.ndarray:
    """Linear in wavelength, so linear resampling reproduces it exactly."""
    return 0.05 + 0.25 * (wl - 400.0) / 2100.0


@pytest.fixture
def patch() -> SpectralEarthPatch:
    bands = load_band_table()
    per_band = np.round(_reflectance(bands.wavelengths_nm) * 10_000).astype(np.int16)
    raw = np.broadcast_to(per_band[:, None, None], (bands.count, 128, 128)).copy()
    rng = np.random.default_rng(0)
    raw[:, 0, 0] = NODATA                                                    # no spectrum at all
    raw[rng.choice(np.arange(20, 180), 20, replace=False), 0, 1] = NODATA    # ~10% gaps: fillable
    raw[rng.choice(bands.count, 101, replace=False), 0, 2] = NODATA          # 50% gaps: unusable
    return SpectralEarthPatch(site_id="0000001", patch_id="00001", raw=raw, bands=bands)


@dataclass
class _RejectPixel:
    """A stand-in screen rejecting one pixel."""

    at: tuple[int, int]
    name: str = "test"

    def keep_mask(self, patch):
        keep = np.ones(patch.raw.shape[1:], dtype=bool)
        keep[self.at] = False
        return keep


def test_nodata_screen_rejects_sparse_spectra_only(patch):
    keep = NodataScreen().keep_mask(patch)
    assert not keep[0, 0] and not keep[0, 2]
    assert keep[0, 1] and keep.sum() == 128 * 128 - 2


def test_chain_ands_screens_and_counts_each_ones_rejections(patch):
    report = ScreenChain([NodataScreen(), _RejectPixel((1, 0)), _RejectPixel((0, 0))]).apply(patch)
    assert report.rejected_by == {"nodata": 2, "test": 1}       # (0,0) was already gone
    assert report.keep.sum() == 128 * 128 - 3
    assert report.usable() and report.keep_fraction > 0.99


def test_to_vendable_lands_on_the_165_band_grid(patch):
    v = patch.to_vendable()
    assert v.normalized_hyperspectral_cube.shape == (165, 128, 128)
    assert v.validity_cube.shape == (165, 128, 128)
    assert np.allclose(v.band_cw_order, DEFAULT_COMMON_WAVELENGTH_GRID)
    expected = _reflectance(np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID))
    assert np.allclose(v.normalized_hyperspectral_cube[:, 64, 64], expected, atol=2e-4)


def test_to_vendable_fills_small_gaps_and_drops_hopeless_pixels(patch):
    v = patch.to_vendable()
    cube, valid = v.normalized_hyperspectral_cube, v.validity_cube
    for r, c in ((0, 0), (0, 2)):
        assert valid[:, r, c].sum() == 0 and not cube[:, r, c].any()
    assert valid[:, 0, 1].all()
    expected = _reflectance(np.asarray(DEFAULT_COMMON_WAVELENGTH_GRID))
    assert np.allclose(cube[:, 0, 1], expected, atol=5e-3)


def test_to_vendable_honours_a_screen_keep_mask(patch):
    report = ScreenChain([NodataScreen(), _RejectPixel((5, 5))]).apply(patch)
    v = patch.to_vendable(keep=report.keep)
    assert v.validity_cube[:, 5, 5].sum() == 0 and not v.normalized_hyperspectral_cube[:, 5, 5].any()
    assert v.validity_cube[:, 5, 6].all()


def test_to_vendable_leaves_the_patch_untouched(patch):
    before = patch.raw.copy()
    patch.to_vendable()
    assert np.array_equal(patch.raw, before)
