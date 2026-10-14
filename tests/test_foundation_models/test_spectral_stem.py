"""The adaptive spectral stem: wavelength encoding, wavelength statistics, cross-attention
compressor and decoder, per-pixel gate.

Parameter counts are pinned to the figures the course derives by hand
(`indradhanu workings/34`, `37`, `39`); a change that moves one must update the course too.
"""

import pytest
import torch
import torch.nn.functional as F

from app.foundation_models.components.pixel_normalization import WavelengthStats
from app.foundation_models.components.spectral_attention import (
    SpectralCrossAttentionCompressor, SpectralCrossAttentionDecoder)
from app.foundation_models.components.spectral_gate import SpectralGate
from app.foundation_models.components.spectral_tokens import wavelength_encoding
from app.models.dataset.spectralearth import load_band_table


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.fixture
def se_wavelengths() -> torch.Tensor:
    return torch.tensor(load_band_table().wavelengths_nm, dtype=torch.float32)


# --- wavelength encoding ------------------------------------------------------------

def test_encoding_matches_course_part_33():
    e = wavelength_encoding(torch.tensor([660.0, 670.0, 1650.0]))
    assert e.shape == (3, 32)
    assert F.cosine_similarity(e[0], e[1], dim=0).item() == pytest.approx(0.671, abs=1e-3)
    assert F.cosine_similarity(e[0], e[2], dim=0).item() == pytest.approx(0.013, abs=1e-3)


def test_encoding_keeps_leading_shape_and_rejects_odd_dim():
    assert wavelength_encoding(torch.zeros(2, 5), dim=8).shape == (2, 5, 8)
    with pytest.raises(ValueError):
        wavelength_encoding(torch.zeros(3), dim=7)


# --- wavelength statistics ------------------------------------------------------------

def test_stats_interpolate_like_course_part_38():
    s = WavelengthStats([870.0, 860.0, 900.0], [0.24, 0.20, 0.30], [0.1, 0.1, 0.2])   # unsorted on purpose
    mean, _ = s.at(torch.tensor([864.0, 850.0, 950.0]))
    assert mean.tolist() == pytest.approx([0.216, 0.20, 0.30])                           # ends held


def test_stats_round_trip_and_live_in_state_dict():
    s = WavelengthStats([400.0, 1000.0, 2500.0], [0.1, 0.3, 0.2], [0.05, 0.1, 0.08])
    x, wl = torch.rand(2, 4, 3, 3), torch.tensor([[450.0, 900, 1500, 2400]] * 2)
    assert torch.allclose(s.denormalize(s.normalize(x, wl), wl), x, atol=1e-6)
    assert set(s.state_dict()) == {"wl", "mean", "std"}


# --- compressor -------------------------------------------------------------------------

def test_compressor_params_and_shape_for_any_band_count(se_wavelengths):
    m = SpectralCrossAttentionCompressor()
    assert _params(m) == 5_345                                       # course part 34
    assert m(torch.randn(2, 202, 4, 4), se_wavelengths).shape == (2, 32, 4, 4)
    assert m(torch.randn(1, 165, 4, 4), torch.linspace(460, 2450, 165)).shape == (1, 32, 4, 4)


def test_masked_band_has_exactly_no_influence(se_wavelengths):
    torch.manual_seed(0)
    m = SpectralCrossAttentionCompressor().eval()
    x = torch.randn(2, 202, 4, 4)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, 100] = False
    poisoned = x.clone()
    poisoned[:, 100] = 99.0
    assert torch.equal(m(x, se_wavelengths, valid), m(poisoned, se_wavelengths, valid))
    assert not torch.allclose(m(x, se_wavelengths), m(poisoned, se_wavelengths))


def test_pixel_with_no_valid_band_outputs_zero_without_nan():
    torch.manual_seed(0)
    m = SpectralCrossAttentionCompressor().eval()
    wl, x = torch.linspace(420, 2440, 50), torch.randn(2, 50, 4, 4)
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[0, :, 0, 0] = False
    y = m(x, wl, valid)
    assert torch.equal(y[0, :, 0, 0], torch.zeros(32))
    assert torch.allclose(y[:, :, 1:], m(x, wl)[:, :, 1:], atol=1e-6)
    y.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_compressor_is_strictly_per_pixel(se_wavelengths):
    torch.manual_seed(0)
    m = SpectralCrossAttentionCompressor().eval()
    x = torch.randn(1, 202, 4, 4)
    changed = x.clone()
    changed[0, :, 2, 2] += 5.0
    diff = (m(x, se_wavelengths) - m(changed, se_wavelengths)).abs().sum(dim=1)[0]
    assert diff[2, 2] > 0 and diff.sum() == diff[2, 2]


def test_per_sample_wavelengths_equal_shared(se_wavelengths):
    torch.manual_seed(0)
    m, x = SpectralCrossAttentionCompressor().eval(), torch.randn(2, 202, 3, 3)
    assert torch.allclose(m(x, se_wavelengths), m(x, se_wavelengths.expand(2, -1)), atol=1e-6)


# --- decoder ----------------------------------------------------------------------------

@pytest.mark.parametrize("bands", [165, 202, 224, 239, 425])
def test_decoder_params_fixed_and_any_band_list(bands):
    m = SpectralCrossAttentionDecoder()
    assert _params(m) == 6_401                                       # course part 37
    assert m(torch.randn(2, 32, 3, 3), torch.linspace(420, 2440, bands)).shape == (2, bands, 3, 3)


def test_decoder_bands_are_independent_questions():
    torch.manual_seed(0)
    m, z = SpectralCrossAttentionDecoder().eval(), torch.randn(2, 32, 3, 3)
    wl = torch.linspace(420, 2440, 30)
    extra = torch.cat([wl[:10], torch.tensor([1000.0]), wl[10:]])
    a, b = m(z, wl), m(z, extra)
    assert torch.allclose(a, torch.cat([b[:, :10], b[:, 11:]], dim=1), atol=1e-6)


# --- gate -------------------------------------------------------------------------------

def test_gate_params_match_course_part_39_and_stay_in_range():
    assert _params(SpectralGate(202)) == 13_162
    assert _params(SpectralGate(165)) == 10_757
    x = torch.randn(2, 202, 4, 4).abs() + 0.1
    ratio = SpectralGate(202)(x) / x
    assert ratio.min() > 0 and ratio.max() < 1
