"""Spectral reduction stem: cheap attention over wavelength bins that still sees every band."""

import pytest
import torch

from app.foundation_models.components.spectral_reduced import (
    SpectralReducedCompressor, WavelengthLinearDecoder, band_widths)
from app.models.dataset.spectralearth import load_band_table

WL = torch.tensor(load_band_table().wavelengths_nm, dtype=torch.float32)


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def test_parameter_counts_do_not_depend_on_band_count():
    assert _params(SpectralReducedCompressor()) == 14_721
    assert _params(WavelengthLinearDecoder()) == 4_257


@pytest.mark.parametrize("wl", [WL, torch.linspace(460, 2450, 165), torch.linspace(400, 2500, 425)])
def test_any_band_list_round_trips_in_shape(wl):
    c, d = SpectralReducedCompressor().eval(), WavelengthLinearDecoder().eval()
    x = torch.randn(1, len(wl), 4, 4)
    assert d(c(x, wl), wl).shape == x.shape


def test_detail_inside_a_bin_is_not_averaged_away():
    """Move value between two bands of one bin, keeping the bin's mean: a plain bin average
    could not see it; per-band weights (fix 1) and the skip path (fix 2) must."""
    torch.manual_seed(0)
    c = SpectralReducedCompressor().eval()
    bins = c.bin_index(WL)
    same = (bins == bins[60]).nonzero().flatten()
    a, b = int(same[0]), int(same[1])
    x = torch.randn(1, 202, 2, 2)
    y = x.clone()
    y[:, a] += 0.5
    y[:, b] -= 0.5
    assert torch.allclose(x[:, same].mean(1), y[:, same].mean(1), atol=1e-6)
    assert (c(x, WL) - c(y, WL)).abs().max() > 1e-4


def test_decoder_gives_each_band_its_own_output():
    d = WavelengthLinearDecoder().eval()
    z = torch.randn(1, 32, 2, 2)
    out = d(z, WL)
    assert out[:, 60].sub(out[:, 61]).abs().max() > 1e-6


def test_dead_band_has_exactly_no_influence_and_dead_pixel_is_zero():
    torch.manual_seed(0)
    c = SpectralReducedCompressor().eval()
    x = torch.randn(2, 202, 3, 3)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, 100] = False
    poisoned = x.clone()
    poisoned[:, 100] = 99.0
    assert torch.equal(c(x, WL, valid), c(poisoned, WL, valid))
    valid[0, :, 0, 0] = False
    z = c(x, WL, valid)
    assert torch.equal(z[0, :, 0, 0], torch.zeros(32)) and torch.isfinite(z).all()


def test_strictly_per_pixel():
    torch.manual_seed(0)
    c = SpectralReducedCompressor().eval()
    x = torch.randn(1, 202, 4, 4)
    y = x.clone()
    y[0, :, 2, 2] += 3.0
    diff = (c(x, WL) - c(y, WL)).abs().sum(dim=1)[0]
    assert diff[2, 2] > 0 and diff.sum() == diff[2, 2]


def test_rejects_per_sample_band_lists():
    with pytest.raises(ValueError):
        SpectralReducedCompressor()(torch.randn(2, 202, 2, 2), WL.expand(2, -1))


# --- band weighting (course part 42; docs/lld/spectral-stem.md) -------------------------------

def test_band_widths_half_gaps_capped_and_in_input_order():
    w = band_widths(torch.tensor([505.0, 500.0, 510.0, 2000.0]))
    assert w.tolist() == [5.0, 5.0, 12.5, 20.0]       # 510 sits beside a capped 1490 nm gap


def test_width_weighting_makes_the_course_toy_density_free():
    """Course part 42: sensor A (500, 505, 2000 nm) and B (500, 2000) see one spectrum, 2 in the
    visible and -1 beyond. One vote per band gives 1.0 vs 0.5; one vote per nm gives 0.5 for both."""
    def mean(wl, v, w):
        return float((v * w).sum() / w.sum())
    a_wl, a_v = torch.tensor([500.0, 505.0, 2000.0]), torch.tensor([2.0, 2.0, -1.0])
    b_wl, b_v = torch.tensor([500.0, 2000.0]), torch.tensor([2.0, -1.0])
    assert (mean(a_wl, a_v, torch.ones(3)), mean(b_wl, b_v, torch.ones(2))) == (1.0, 0.5)
    a_w, b_w = torch.tensor([5.0, 5.0, 10.0]), torch.tensor([10.0, 10.0])   # the course's widths
    assert mean(a_wl, a_v, a_w) == mean(b_wl, b_v, b_w) == 0.5


def test_width_equals_count_on_an_even_grid():
    """Equal widths cancel, so on evenly spaced bands the two weightings agree exactly."""
    torch.manual_seed(0)
    count = SpectralReducedCompressor(band_weighting="count").eval()
    width = SpectralReducedCompressor(band_weighting="width").eval()
    width.load_state_dict(count.state_dict())
    wl, x = torch.arange(500.0, 1500.0, 10.0), torch.randn(1, 100, 2, 2)
    assert torch.allclose(count(x, wl), width(x, wl), atol=1e-5)


def test_width_weighting_cuts_sensitivity_to_one_regions_density():
    torch.manual_seed(0)
    count = SpectralReducedCompressor(band_weighting="count").eval()
    width = SpectralReducedCompressor(band_weighting="width").eval()
    width.load_state_dict(count.state_dict())
    dense = torch.cat([torch.arange(500.0, 700.0, 5.0), torch.arange(1000.0, 1400.0, 10.0)])
    even = torch.cat([torch.arange(500.0, 700.0, 10.0), torch.arange(1000.0, 1400.0, 10.0)])
    spec = lambda wl: torch.where(wl < 800, 2.0, -1.0)[None, :, None, None]
    shift = lambda m: float((m(spec(dense), dense) - m(spec(even), even)).abs().mean())
    assert shift(width) < 0.5 * shift(count)


def test_unknown_band_weighting_is_rejected():
    with pytest.raises(ValueError):
        SpectralReducedCompressor(band_weighting="per_band")
