"""Fine-tuning pieces: segmentation targets and loss, the trained-wavelength record, the segmentation head,
the loader-worker transform and the sensor mixer (docs/proposals/fine-tuning-and-segmentation.md)."""

import io

import numpy as np
import pytest
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.trainers.finetune_trainer import _AllSourcesLoader, _MixedLoader, gradient_l1, prepare_sample
from app.utils.segmentation.targets import IoUTally, LabelPolicy, enmap_targets, masked_bce_dice

WL = torch.linspace(420, 2440, 202)
STATS = (WL.tolist(), [0.2] * 202, [0.1] * 202)
SMALL = dict(embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)
H = W = 8


def _labels():
    lab = {k: np.zeros((1, H, W), np.uint8) for k in
           ("label_cloud", "label_cirrus", "label_haze", "label_cloud_shadow", "label_snow", "label_classes")}
    lab["label_classes"][0] = 1
    lab["label_classes"][0, :2] = 2           # water rows
    lab["label_classes"][0, 7] = 3            # provider no-data row (3; 0 is EnMAP's cloud class)
    lab["label_cloud"][0, 4, 4] = 1
    lab["label_cirrus"][0, 5, 5], lab["label_cirrus"][0, 6, 6] = 1, 3
    return lab


def test_targets_classes_cirrus_level_and_nodata():
    t, k = enmap_targets(_labels(), {"mean_ground_elevation_m": 300.0, "scene_wv": 2000.0})
    assert t[0, :2].all() and not t[0, 2:].any()                    # water
    assert t[1, 4, 4] == 1 and t[2, 6, 6] == 1                      # cloud; cirrus 3 is thin cloud
    assert not k[2, 5, 5] and k[2, 4, 4]                            # cirrus 1: thin cloud unknown, not clear
    assert not k[0, 4, 4] and not k[0, 6, 6]                        # no water truth under cloud / thick cirrus
    assert not k[:, 7].any() and k[1, :7].all()


def test_class_zero_is_cloud_and_snow_wins():
    lab = _labels()
    lab["label_classes"][0, 3] = 0                                   # EnMAP's cloud class
    lab["label_cloud"][0, 2, 2] = 1; lab["label_snow"][0, 2, 2] = 1
    t, k = enmap_targets(lab, {"scene_wv": 2000.0})
    assert t[1, 3].all() and k[1, 3].all() and not k[[0, 2, 3, 4], 3].any()
    assert not k[1, 2, 2] and k[4, 2, 2] and t[4, 2, 2] == 1


def test_dry_or_high_scenes_switch_off_only_cloud_labels():
    for meta in ({"scene_wv": 600.0}, {"mean_ground_elevation_m": 4200.0, "scene_wv": 2000.0}):
        _, k = enmap_targets(_labels(), meta)                      # default policy: wv < 750 or elevation >= 3500
        assert not k[1:3].any() and k[0, 0].all() and k[3, 0].all() and k[4, 0].all()
    _, k = enmap_targets(_labels(), {"scene_wv": 600.0}, LabelPolicy(low_water_vapour=None))
    assert k[1, 0].all()


def test_loss_ignores_unknown_pixels_and_iou_counts_known_only():
    t, k = (torch.from_numpy(a)[None] for a in enmap_targets(_labels(), {"scene_wv": 2000.0}))
    wrong_where_unknown = torch.where(k, t * 20 - 10, 10 - t * 20)   # perfect on known pixels only
    assert masked_bce_dice(wrong_where_unknown, t, k) < 0.05
    tally = IoUTally(); tally.add(wrong_where_unknown, t, k)
    assert tally.iou()["water"] == 1.0 and tally.iou()["snow"] is None


def test_trained_wavelength_record_grows_and_survives_a_reload():
    m = SpectralStemSegFormerMAE("reduced_attention", *STATS, **SMALL)
    assert torch.equal(m.trained_wl, torch.sort(m.stats.wl).values)
    on_list = float(m.stats.wl[50])
    assert m.extend_trained_wavelengths([1450.0, 1450.2, on_list + 0.1]) == 1   # one new; already on the list
    assert bool(m.trained_band_mask([1450.0])[0])
    sd = m.state_dict()
    fresh = SpectralStemSegFormerMAE("reduced_attention", *STATS, **SMALL)
    fresh.load_state_dict(sd)
    assert torch.equal(fresh.trained_wl, m.trained_wl)
    old = {k: v for k, v in sd.items() if k != "trained_wl"}             # a pre-2026-10-16 checkpoint
    fresh.load_state_dict(old)
    assert torch.equal(fresh.trained_wl, torch.sort(fresh.stats.wl).values)


def test_segmentation_head_is_optional_and_leaves_reconstruction_alone():
    torch.manual_seed(0)
    plain = SpectralStemSegFormerMAE("reduced_attention", *STATS, **SMALL).eval()
    seg = SpectralStemSegFormerMAE("reduced_attention", *STATS, seg_classes=5, seg_decoder_dim=16, **SMALL).eval()
    missing, unexpected = seg.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("seg_head.") for k in missing)
    x = torch.rand(1, 202, 64, 64) * 0.3
    with torch.no_grad():
        assert torch.equal(plain(x, WL), seg(x, WL))
        assert seg.segment(x, WL).shape == (1, 5, 64, 64)
    with pytest.raises(ValueError):
        plain.segment(x, WL)


def test_prepare_sample_makes_targets_and_drops_what_does_not_collate():
    lab = _labels()
    sample = {"pixels.npy": np.zeros((3, H, W), np.int16), "wavelengths.npy": np.array([500, 600, 700], np.float32),
              "meta.json": {"mean_ground_elevation_m": 100.0, "cover_pct": {"cloud": None}}}
    sample.update({f"{k}.npy": v for k, v in lab.items()})
    out = prepare_sample(dict(sample), labels="enmap")
    assert out["seg_targets.npy"].shape == (5, H, W) and out["seg_known.npy"].dtype == bool
    assert "meta.json" not in out and not any(k.startswith("label_") for k in out)
    plain = prepare_sample(dict(sample), labels=None)
    assert "seg_targets.npy" not in plain and "meta.json" not in plain


def test_mixed_loader_tags_sources_follows_shares_and_restarts():
    loaders = {"a": [{"x": 1}, {"x": 2}], "b": [{"x": 3}]}
    it = iter(_MixedLoader(loaders, {"a": 3.0, "b": 1.0}, seed=0))
    names = [next(it)["_source"] for _ in range(400)]
    share_a = names.count("a") / len(names)
    assert 0.65 < share_a < 0.85 and set(names) == {"a", "b"}             # endless: both restart when exhausted


def test_forward_mixed_matches_forward_per_group():
    """In eval mode (stored BatchNorm statistics) a mixed pass equals one forward per band list."""
    torch.manual_seed(0)
    m = SpectralStemSegFormerMAE("reduced_attention", *STATS, **SMALL).eval()
    a_wl, b_wl = WL[::2], WL[1::3] + 1.5                       # two different band lists
    a, b = torch.rand(2, len(a_wl), 64, 64) * 0.3, torch.rand(1, len(b_wl), 64, 64) * 0.3
    groups = [(a, a_wl, None, None, None), (b, b_wl, None, None, WL)]
    with torch.no_grad():
        out_a, out_b = m.forward_mixed(groups)
        assert torch.allclose(out_a, m(a, a_wl), atol=1e-5)
        assert torch.allclose(out_b, m(b, b_wl, out_wavelengths_nm=WL), atol=1e-5)
    assert out_b.shape == (1, len(WL), 64, 64)


def test_all_sources_loader_puts_every_source_in_every_batch_and_restarts():
    it = iter(_AllSourcesLoader({"a": [{"k": 1}, {"k": 2}], "b": [{"k": 9}]}))
    batches = [next(it) for _ in range(3)]
    assert all([name for name, _ in b["_groups"]] == ["a", "b"] for b in batches)
    assert [b["_groups"][0][1]["k"] for b in batches] == [1, 2, 1]       # a restarts after its 2 batches
    assert [b["_groups"][1][1]["k"] for b in batches] == [9, 9, 9]


def test_mixed_batches_not_in_probe():
    from app.models.training.training_config import SpectralStemFinetuneConfig
    src = [{"name": "enmap", "train_dir": "t", "test_dir": "v", "share": 1.0}]
    assert SpectralStemFinetuneConfig(sources=src, mode="adapt", mixed_batches=True).mixed_batches
    assert SpectralStemFinetuneConfig(sources=src, mode="joint", mixed_batches=True).mixed_batches
    with pytest.raises(ValueError):
        SpectralStemFinetuneConfig(sources=src, mode="probe", mixed_batches=True)


def test_gradient_l1_by_hand():
    """x = 0, 0, 1 and x_hat = 0, 0.2, 0.9 along one row: differences (0, 1) vs (0.2, 0.7) -> |.| 0.2, 0.3."""
    x, x_hat = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3), torch.tensor([0.0, 0.2, 0.9]).view(1, 1, 1, 3)
    ones = torch.ones(1, 1, 1, 3)
    assert gradient_l1(x_hat, x, ones.bool(), ones).item() == pytest.approx(0.25)
    assert gradient_l1(x, x, ones.bool(), ones).item() == 0.0
    assert gradient_l1(x_hat, x, ones.bool(), torch.tensor([1.0, 1.0, 0.0]).view(1, 1, 1, 3)).item() == pytest.approx(0.2)
    assert gradient_l1(x_hat, x, torch.tensor([False, True, True]).view(1, 1, 1, 3), ones).item() == pytest.approx(0.3)


def test_grad_loss_needs_mixed_batches():
    from app.models.training.training_config import SpectralStemFinetuneConfig
    src = [{"name": "enmap", "train_dir": "t", "test_dir": "v", "share": 1.0}]
    with pytest.raises(ValueError):
        SpectralStemFinetuneConfig(sources=src, mode="adapt", grad_loss_weight=1.0)
    assert SpectralStemFinetuneConfig(sources=src, mode="adapt", mixed_batches=True, grad_loss_weight=1.0).grad_loss_weight == 1.0


def test_segment_mixed_matches_segment_per_group():
    torch.manual_seed(0)
    m = SpectralStemSegFormerMAE("reduced_attention", *STATS, seg_classes=5, seg_decoder_dim=16, **SMALL).eval()
    a_wl, b_wl = WL[::2], WL[1::3] + 1.5
    a, b = torch.rand(2, len(a_wl), 64, 64) * 0.3, torch.rand(1, len(b_wl), 64, 64) * 0.3
    with torch.no_grad():
        la, lb = m.segment_mixed([(a, a_wl, None), (b, b_wl, None)])
        assert torch.allclose(la, m.segment(a, a_wl), atol=1e-5) and torch.allclose(lb, m.segment(b, b_wl), atol=1e-5)
    assert la.shape == (2, 5, 64, 64)


def test_implant_sample_keeps_clean_target_and_marks_near_pixels(tmp_path):
    from app.foundation_models.trainers.finetune_trainer import implant_sample
    rng = np.random.default_rng(0)
    wl = np.linspace(460, 2400, 40)
    # shared brightness structure plus a little band noise, as in a real scene (pure band noise is all "foreign")
    pix = (2000 + 500 * rng.standard_normal((1, 128, 128)) + 20 * rng.standard_normal((40, 128, 128))).astype(np.int16)
    real = {"wavelengths_nm": {"s": wl.tolist()},
            "objects": [{"scene": "s", "px": 2, "z": z, "foreign": 0.01, "d": list(0.02 * np.sin(wl / 200))} for z in (0.3, 0.7, 1.4)]}
    cfg = {"prob": 1.0, "max_objects": 3, "psf_sigma": 1.4, "sources": ["signature"], "seed": 1, "real": real,
           "lib_cache": str(tmp_path)}
    out = implant_sample({"pixels.npy": pix.copy(), "wavelengths.npy": wl, "__key__": "t1"}, cfg)
    assert np.array_equal(out["clean.npy"], pix)                       # the target is the untouched tile
    assert out["implant.npy"].any() and (out["pixels.npy"] != pix).any()
    from scipy import ndimage                                         # the PSF tail reaches a little past the
    dist = ndimage.distance_transform_edt(out["implant.npy"] == 0)    # 3-px weighting zone (measured 2.8 px), no further
    assert dist[(out["pixels.npy"] != pix).any(0)].max() <= 3.0
    again = implant_sample({"pixels.npy": pix.copy(), "wavelengths.npy": wl, "__key__": "t1"}, cfg)
    assert np.array_equal(again["pixels.npy"], out["pixels.npy"])      # seeded by the key: same implants each epoch
    off = implant_sample({"pixels.npy": pix.copy(), "wavelengths.npy": wl, "__key__": "t1"}, {**cfg, "prob": 0.0})
    assert np.array_equal(off["pixels.npy"], pix) and not off["implant.npy"].any()   # keys present, nothing implanted


def test_pseudo_targets_guards_by_hand():
    """Four pixels the teacher calls: cloud on a flat white spectrum and on a reddish one - both kept (no cloud guard:
    it rejected most real clouds); water with NIR 0.03 - kept; water with NIR 0.3 - becomes NOT water."""
    from app.utils.segmentation.targets import pseudo_targets
    wl = np.array([480.0, 560.0, 660.0, 860.0])
    refl = np.zeros((4, 1, 4))
    refl[:, 0, 0] = 0.5
    refl[:, 0, 1] = [0.25, 0.3, 0.35, 0.5]
    refl[:, 0, 2] = [0.05, 0.04, 0.03, 0.03]
    refl[:, 0, 3] = [0.2, 0.2, 0.2, 0.3]
    probs = np.zeros((5, 1, 4), np.uint8)
    probs[1, 0, :2] = 250                                 # teacher: cloud at 0, 1
    probs[0, 0, 2:] = 250                                 # teacher: water at 2, 3
    t, k = pseudo_targets(probs, refl, wl, np.ones((1, 4), bool))
    assert t[1, 0].tolist() == [1, 1, 0, 0] and k[1, 0].all()          # both teacher clouds taught as cloud
    assert t[0, 0].tolist() == [0, 0, 1, 0] and k[0, 0].all()          # bright "water" -> known not-water
    assert not k[2:].any()                                             # thin cloud, shadow, snow untaught
    probs[1, 0, 0] = 128                                               # an unsure teacher teaches nothing
    assert not pseudo_targets(probs, refl, wl, np.ones((1, 4), bool))[1][1, 0, 0]


def test_class_weights_by_hand():
    from app.utils.segmentation.targets import masked_bce_dice
    logits = torch.zeros(1, 2, 1, 1)                       # p = 0.5 for both classes
    targets = torch.tensor([[[[1.0]], [[0.0]]]])
    known = torch.ones(1, 2, 1, 1, dtype=torch.bool)
    # per class: BCE = ln 2 = 0.6931; Dice class 0: 1 - (2*0.5 + 1) / (0.5 + 1 + 1) = 0.2, class 1: 1 - 1 / 1.5 = 0.3333
    plain = masked_bce_dice(logits, targets, known)
    assert plain.item() == pytest.approx(0.6931 + (0.2 + 0.3333) / 2, abs=1e-3)
    w = masked_bce_dice(logits, targets, known, class_weights=[1.0, 3.0])
    assert w.item() == pytest.approx(0.6931 + (0.2 * 1 + 0.3333 * 3) / 4, abs=1e-3)
