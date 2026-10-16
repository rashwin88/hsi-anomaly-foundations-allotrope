"""Fine-tuning pieces: segmentation targets and loss, the trained-wavelength record, the segmentation head,
the loader-worker transform and the sensor mixer (docs/proposals/fine-tuning-and-segmentation.md)."""

import io

import numpy as np
import pytest
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.trainers.finetune_trainer import _MixedLoader, prepare_sample
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
