"""SpectralStemSegFormerMAE: the three experiment variants around one shared SegFormer."""

import pytest
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking

WL = torch.linspace(420, 2440, 202)
STATS = (WL.tolist(), [0.2] * 202, [0.1] * 202)
SMALL = dict(embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)


def _params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("stem,stem_in,stem_out", [
    ("linear", 6_496, 6_666), ("gate", 19_658, 6_666), ("cross_attention", 5_345, 6_401)])
def test_stems_differ_only_in_the_stem(stem, stem_in, stem_out):
    m = SpectralStemSegFormerMAE(stem, *STATS)
    assert (_params(m.spectral_in), _params(m.spectral_out)) == (stem_in, stem_out)
    shared = _params(m) - stem_in - stem_out
    assert shared == _params(SpectralStemSegFormerMAE("linear", *STATS)) - 6_496 - 6_666


@pytest.mark.parametrize("stem", ["linear", "gate", "cross_attention"])
def test_forward_backward_with_token_removal_and_a_dead_band(stem):
    torch.manual_seed(0)
    m = SpectralStemSegFormerMAE(stem, *STATS, **SMALL)
    x = torch.rand(2, 202, 64, 64) * 0.4
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, 50] = False
    keep, _ = TokenMasking.generate_prediction_mask(torch.ones(2, 256), 0.65, "cpu")
    y = m(x, WL, valid, keep)
    y.abs().mean().backward()
    assert y.shape == x.shape and torch.isfinite(y).all()
    assert all(p.grad is not None for p in m.spectral_in.parameters())


def test_cross_attention_reads_another_band_list_and_fixed_stems_refuse():
    grid = torch.linspace(460, 2450, 165)
    c = SpectralStemSegFormerMAE("cross_attention", *STATS, **SMALL).eval()
    assert c(torch.rand(1, 165, 64, 64), grid).shape == (1, 165, 64, 64)
    for stem in ("linear", "gate"):
        with pytest.raises(ValueError):
            SpectralStemSegFormerMAE(stem, *STATS, **SMALL)(torch.rand(1, 165, 64, 64), grid)


def test_dead_voxel_value_never_reaches_any_stem():
    torch.manual_seed(0)
    x = torch.rand(1, 202, 64, 64) * 0.4
    valid = torch.ones_like(x, dtype=torch.bool)
    valid[:, 7] = False
    poisoned = x.clone()
    poisoned[:, 7] = 50.0
    for stem in ("linear", "gate", "cross_attention"):
        m = SpectralStemSegFormerMAE(stem, *STATS, **SMALL).eval()
        assert torch.equal(m(x, WL, valid), m(poisoned, WL, valid)), stem


def test_unknown_stem_rejected():
    with pytest.raises(ValueError):
        SpectralStemSegFormerMAE("selfattention", *STATS)
