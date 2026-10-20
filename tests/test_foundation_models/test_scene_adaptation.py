"""Gated per-scene adaptation (app/foundation_models/scene_adaptation.py)."""
import numpy as np
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.scene_adaptation import _valid_fraction, adapt, gate_mask

WL = torch.linspace(420, 2440, 202)
STATS = (WL.tolist(), [0.2] * 202, [0.1] * 202)
SMALL = dict(embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)


def test_gate_mask_by_hand():
    score = np.arange(100, dtype=float).reshape(10, 10)
    valid = np.ones((10, 10), bool)
    g = gate_mask(score, valid, pct=1.0, halo=0)
    assert g.sum() == 1 and g[9, 9]                      # the top 1% of 100 pixels is the single highest
    assert gate_mask(score, valid, pct=1.0, halo=1).sum() == 3      # corner (9,9) + its 4-neighbours (8,9), (9,8)
    assert not gate_mask(score, valid, pct=0, halo=2).any()


def test_valid_fraction_by_hand():
    v = np.zeros((4, 4), bool)
    v[:2, :2] = True
    assert np.allclose(_valid_fraction(v, 2), [[1, .5, 0], [.5, .25, 0], [0, 0, 0]])


def scene_and_model(seed=0):
    torch.manual_seed(seed)
    m = SpectralStemSegFormerMAE("reduced_attention", *STATS, seg_classes=5, seg_decoder_dim=16, **SMALL)
    wl = WL[::4].numpy()
    cube = (np.random.default_rng(seed).random((len(wl), 160, 160)) * 0.3).astype(np.float32)
    return m, cube, wl, np.ones((160, 160), bool)


def test_adapt_lowers_its_loss_and_leaves_the_seg_head_alone():
    m, cube, wl, valid = scene_and_model()
    seg = {k: v.clone() for k, v in m.state_dict().items() if k.startswith("seg_head.")}
    losses = adapt(m, cube, wl, valid, np.zeros_like(valid), steps=30, lr=1e-3, batch=2)
    assert len(losses) == 30 and np.mean(losses[-5:]) < np.mean(losses[:5])
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in seg.items())


def test_a_fully_gated_scene_teaches_nothing():
    m, cube, wl, valid = scene_and_model()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert adapt(m, cube, wl, valid, np.ones_like(valid), steps=5, lr=1e-3, batch=2) == []
    assert all(torch.equal(v, m.state_dict()[k]) for k, v in before.items())
