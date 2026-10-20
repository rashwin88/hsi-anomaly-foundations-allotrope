"""The uncertainty head (FT06, docs/proposals/uncertainty-head.md): optional, leaves reconstruction alone, sees only
the masked context, and its Laplace loss is lowest where b equals the error."""

import pytest
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.trainers.finetune_trainer import spearman

WL = torch.linspace(420, 2440, 202)
STATS = (WL.tolist(), [0.2] * 202, [0.1] * 202)
SMALL = dict(embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)


def models():
    torch.manual_seed(0)
    plain = SpectralStemSegFormerMAE("reduced_attention", *STATS, seg_classes=5, seg_decoder_dim=16, **SMALL).eval()
    scaled = SpectralStemSegFormerMAE("reduced_attention", *STATS, seg_classes=5, seg_decoder_dim=16,
                                      scale_decoder_dim=16, **SMALL).eval()
    missing, unexpected = scaled.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("scale_head.") for k in missing)
    return plain, scaled


def test_head_is_optional_and_leaves_reconstruction_alone():
    plain, scaled = models()
    x = torch.rand(2, 202, 64, 64) * 0.3
    with torch.no_grad():
        recon, log_b = scaled(x, WL, with_scale=True)
        assert torch.equal(recon, plain(x, WL)) and log_b.shape == (2, 1, 64, 64)
        out, scales = scaled.forward_mixed([(x, WL, None, None, None)], with_scale=True)
        assert torch.allclose(out[0], recon, atol=1e-5) and torch.allclose(scales[0], log_b, atol=1e-5)


def test_hidden_pixels_value_cannot_reach_its_own_scale():
    """With a token removed, changing that token's pixels leaves its predicted log b unchanged."""
    _, scaled = models()
    x = torch.rand(1, 202, 64, 64) * 0.3
    tokens = (64 // 4) ** 2                                    # stage-1 stride 4
    keep = torch.ones(1, tokens)
    keep[0, 0] = 0                                             # hide the top-left 4x4 token
    y = x.clone()
    y[:, :, :4, :4] += 0.5
    with torch.no_grad():
        _, a = scaled(x, WL, keep_mask=keep, with_scale=True)
        _, b = scaled(y, WL, keep_mask=keep, with_scale=True)
    assert torch.allclose(a[..., :4, :4], b[..., :4, :4], atol=1e-5)


def test_laplace_loss_is_lowest_where_b_equals_the_error():
    e = torch.tensor(0.3)
    nll = lambda b: e / b + torch.log(b)
    # e/b + log b at b = 0.15, 0.3, 0.6: 2 + log 0.15 = 0.103, 1 + log 0.3 = -0.204, 0.5 + log 0.6 = -0.011
    assert nll(torch.tensor(0.3)) < nll(torch.tensor(0.15)) and nll(torch.tensor(0.3)) < nll(torch.tensor(0.6))
    assert float(nll(torch.tensor(0.3))) == pytest.approx(1 + float(torch.log(e)))


def test_spearman():
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    assert spearman(a, torch.tensor([1.0, 3.0, 2.0, 4.0])) == pytest.approx(0.8)    # 1 - 6*2/(4*15)


def test_mode_scale_needs_a_head():
    from app.models.training.training_config import SpectralStemFinetuneConfig
    src = [{"name": "enmap", "train_dir": "t", "test_dir": "v", "share": 1.0}]
    assert SpectralStemFinetuneConfig(sources=src, mode="scale", scale_decoder_dim=64, mixed_batches=True).mode == "scale"
    with pytest.raises(ValueError):
        SpectralStemFinetuneConfig(sources=src, mode="scale")


def test_inferencer_scale_pass_matches_plain_reconstruction(tmp_path):
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
    from app.models.training.inference_config import InferenceConfig
    from app.models.training.training_config import SpectralStemFinetuneConfig
    _, scaled = models()
    cfg = SpectralStemFinetuneConfig(sources=[{"name": "e", "train_dir": "t", "test_dir": "v", "share": 1.0}],
                                     stem="reduced_attention", seg_classes=5, seg_decoder_dim=16, scale_decoder_dim=16,
                                     embed_dims=[8, 16, 32, 64], num_heads=[1, 1, 2, 2], decoder_dim=32).model_dump()
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": scaled.state_dict(), "config": {"model_config": cfg}}, path)
    inf = SpectralStemInferencer(InferenceConfig(foundation_model_name="spectral_stem_finetune", model_config=cfg,
                                                 checkpoint_path=str(path), patch_size=64, stride=32,
                                                 inference_batch_size=4))
    inf.set_wavelengths(WL)
    scene, mask = torch.rand(202, 96, 96) * 0.3, torch.ones(1, 96, 96)
    torch.manual_seed(1)
    recon, log_b = inf.predict_full_scene_with_scale(scene, mask)
    torch.manual_seed(1)
    plain = inf.predict_full_scene(scene, mask)
    assert log_b.shape == (96, 96) and torch.allclose(recon, plain, atol=1e-5)
