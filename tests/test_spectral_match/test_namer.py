"""Spectral namer: class map, generator, model (app/spectral_match/namer, docs/proposals/spectral-namer.md)."""

import torch

from app.spectral_match.namer.bank import LibraryBank
from app.spectral_match.namer.classes import MANMADE_CLASSES, class_of
from app.spectral_match.namer.generator import plant, ring_and_noise
from app.spectral_match.namer.model import SpectralNamer


def test_class_map_dissolves_roofing_by_material():
    assert class_of("natural.vegetation", "Grass") == "natural"
    assert class_of("manmade.pavement", "Asphalt") == "concrete_asphalt"
    assert class_of("manmade.roofing", "Asphalt_Shingle") == "concrete_asphalt"
    assert class_of("manmade.roofing", "Cedar_Shake") == "wood_paper"
    assert class_of("manmade.roofing", "Fiberglass Grn roofing") == "plastic"
    assert class_of("manmade.glass", "Window glass") is None


def test_ring_mean_of_a_constant_tile_is_the_constant_and_noise_zero():
    px = torch.full((1, 3, 20, 20), 0.3)
    ring, count, sigma = ring_and_noise(px, torch.ones_like(px, dtype=torch.bool))
    assert torch.allclose(ring[0, :, 10, 10], torch.full((3,), 0.3))
    assert count[0, 0, 10, 10] == 15 * 15 - 5 * 5
    assert torch.allclose(sigma, torch.zeros(1, 3), atol=1e-6)


def _bank(n_bands=8):
    cls = torch.tensor([-1, -1, 0, 0, 1, 1, 2])
    test = torch.tensor([False, False, False, True, False, True, False])
    return LibraryBank(torch.rand(7, n_bands), cls, test, [str(i) for i in range(7)], holdout_class=2)


def test_plant_keeps_test_and_open_set_spectra_out_of_training():
    g = torch.Generator().manual_seed(0)
    px = torch.rand(2, 8, 24, 24) * 0.1 + 0.2
    s = plant(px, torch.ones_like(px, dtype=torch.bool), _bank(), per_tile=200, kinds=(1.0, 0.0, 0.0), generator=g)
    assert set(s["entry"].tolist()) <= {2, 4}                               # train split, not the held-out class
    assert (s["manmade"] == 1).all() and ((s["fill"] >= 0.05) & (s["fill"] <= 1)).all()
    t = plant(px, torch.ones_like(px, dtype=torch.bool), _bank(), per_tile=200, kinds=(1.0, 0.0, 0.0), split="test",
              fixed_fill=0.25, generator=g)
    assert set(t["entry"].tolist()) <= {3, 5, 6} and torch.allclose(t["fill"], torch.tensor(0.25))


def test_full_fill_without_gain_or_noise_is_the_library_spectrum():
    px = torch.full((1, 8, 24, 24), 0.2)                                    # flat tile: zero noise
    bank = _bank()
    s = plant(px, torch.ones_like(px, dtype=torch.bool), bank, per_tile=10, gain=0.0, kinds=(1.0, 0.0, 0.0),
              fixed_fill=1.0)
    assert torch.allclose(s["x"], bank.refl[s["entry"]], atol=1e-6)


def _namer():
    wl = torch.linspace(400, 2500, 50)
    return SpectralNamer(wl.tolist(), [0.2] * 50, [0.1] * 50, compressed_channels=16, token_dim=32, attention_heads=2,
                         hidden=32, embed=16)


def test_class_scores_leave_out_the_planted_entry():
    m = _namer()
    z = torch.nn.functional.normalize(torch.randn(3, 16), dim=1)
    e = torch.cat([z[:1], torch.nn.functional.normalize(torch.randn(2, 16), dim=1)])
    cls = torch.tensor([0, 1, -1])
    full = m.class_scores(z, e, cls, len(MANMADE_CLASSES))
    out = m.class_scores(z, e, cls, len(MANMADE_CLASSES), exclude=torch.tensor([0, -1, -1]))
    assert torch.isinf(out[0, 0]) and torch.equal(out[1:], full[1:])     # class 0 had only the excluded entry
    assert torch.isinf(full[:, 2:]).all()                                  # classes with no entry


def test_namer_reads_any_band_list_and_loads_a_stem_checkpoint():
    m = _namer()
    for n in (60, 37):
        wl = torch.linspace(420, 2400, n)
        z, logit, fill = m.forward_pixel(torch.rand(5, n), torch.rand(5, n), wl)
        assert z.shape == (5, 16) and logit.shape == (5,) and ((fill >= 0) & (fill <= 1)).all()
        assert m.forward_library(torch.rand(4, n), wl).shape == (4, 16)
    state = {"stats.wl": m.stats.wl, "stats.mean": m.stats.mean, "stats.std": m.stats.std,
             **{f"spectral_in.{k}": v for k, v in m.stem.state_dict().items()},
             **{f"norm.{k}": v for k, v in m.norm.state_dict().items()}}
    m2 = SpectralNamer.from_spectral_stem(state, compressed_channels=16, token_dim=32, attention_heads=2, hidden=32,
                                          embed=16)
    assert torch.equal(m2.stem.queries, m.stem.queries)


def test_ring_means_match_the_generator_on_a_valid_tile():
    from app.spectral_match.namer.infer import ring_means
    cube = torch.rand(4, 20, 20)
    ring, _, _ = ring_and_noise(cube[None], torch.ones(1, 4, 20, 20, dtype=torch.bool))
    rows, cols = torch.tensor([10, 7]), torch.tensor([9, 12])
    got = ring_means(cube.numpy(), torch.ones(4, 20, 20, dtype=torch.bool).numpy(), rows.numpy(), cols.numpy())
    assert torch.allclose(torch.from_numpy(got), ring[0, :, rows, cols].T, atol=1e-5)


def test_level1_by_matching_compares_man_made_with_natural():
    m = _namer()
    scores = torch.tensor([[2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.0],      # man-made match strong, natural weak
                           [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 4.0]])  # natural strong
    logit = m.level1_logit(scores)
    assert logit[0] > 0 > logit[1]
    assert torch.isclose(logit[1], torch.logsumexp(scores[1, :7], 0) - 4.0)


def test_warp_shifts_a_spectrum_along_wavelength():
    from app.spectral_match.namer.generator import warp
    wl = torch.linspace(600, 500, 11)                       # 10 nm bands, descending (any order works)
    t = (wl / 1000)[None].repeat(2, 1)                      # linear in wavelength
    out = warp(t, wl, torch.tensor([10.0, 0.0]), torch.tensor([0.0, 0.0]))
    assert torch.allclose(out[0], (wl + 10).clamp(max=600) / 1000, atol=1e-6)
    assert torch.allclose(out[1], wl / 1000, atol=1e-6)


def test_mixing_stays_within_the_class_and_training_pool():
    from app.spectral_match.namer.generator import mix_within_class
    refl = torch.eye(5)
    labels = torch.tensor([0, 0, 1, 1, -1])
    pools = {0: torch.tensor([0, 1]), 1: torch.tensor([2, 3]), -1: torch.tensor([4])}
    out = mix_within_class(torch.tensor([0, 2, 2, 4]), labels, refl, pools, p=1.0,
                           generator=torch.Generator().manual_seed(0))
    assert (out[0, 2:] == 0).all() and (out[1, :2] == 0).all() and (out[1, 4] == 0)
    assert torch.equal(out[3], refl[4])                     # a one-spectrum pool has no partner: unchanged


def test_dropout_keeps_the_old_checkpoint_layout():
    m0, m1 = _namer(), SpectralNamer(torch.linspace(400, 2500, 50).tolist(), [0.2] * 50, [0.1] * 50,
                                     compressed_channels=16, token_dim=32, attention_heads=2, hidden=32, embed=16,
                                     dropout=0.1)
    assert "pixel.4.weight" in m0.state_dict() and "pixel.6.weight" in m1.state_dict()
