"""FT07 / FT08: training-loss gating and block masking (finetune_trainer.gate_pixels, _build_prediction_mask)."""
import types

import pytest
import torch

from app.foundation_models.trainers.finetune_trainer import FinetuneTrainer, gate_pixels


def test_gate_pixels_by_hand():
    x = torch.zeros(1, 2, 10, 10)
    x_hat = torch.zeros(1, 2, 10, 10)
    x_hat[0, :, 5, 5] = 1.0                       # one pixel with error 1, the other 99 with error 0
    bv = torch.ones(1, 2, 10, 10, dtype=torch.bool)
    lm = torch.ones(1, 1, 10, 10)
    g = gate_pixels(x_hat, x, bv, lm, pct=1.0, halo=0)
    assert g.sum() == 1 and g[0, 0, 5, 5]         # top 1% of 100 loss pixels = the one odd pixel
    assert gate_pixels(x_hat, x, bv, lm, pct=1.0, halo=1).sum() == 9      # 3 x 3 square around it
    lm[0, 0, 5, 5] = 0                            # not a loss pixel -> cannot be gated
    assert not gate_pixels(x_hat, x, bv, lm, pct=1.0, halo=0)[0, 0, 5, 5]


def block_trainer(block):
    t = FinetuneTrainer.__new__(FinetuneTrainer)
    t.config = types.SimpleNamespace(model_config_=types.SimpleNamespace(mask_block=block))
    t.device = torch.device("cpu")
    return t


def test_block_masks_hide_whole_blocks_at_the_ratio():
    torch.manual_seed(0)
    pixel = torch.ones(4, 1, 128, 128)
    keep, pred = block_trainer(2)._build_prediction_mask(pixel, 0.65)
    grid = pred.reshape(4, 32, 32)
    blocks = grid.reshape(4, 16, 2, 16, 2).permute(0, 1, 3, 2, 4).reshape(4, 256, 4)
    assert ((blocks.sum(-1) == 0) | (blocks.sum(-1) == 4)).all()          # every 2 x 2 block all-or-nothing
    assert pred.sum(1).tolist() == [int(256 * 0.65) * 4] * 4              # 166 of 256 blocks, 4 tokens each
    assert torch.equal(keep + pred, torch.ones_like(keep))


def test_block_masks_leave_invalid_tokens_kept():
    torch.manual_seed(0)
    pixel = torch.ones(1, 1, 128, 128)
    pixel[..., :, :64] = 0                                                # left half nodata
    keep, pred = block_trainer(2)._build_prediction_mask(pixel, 0.65)
    assert pred.reshape(32, 32)[:, :16].sum() == 0 and keep.reshape(32, 32)[:, :16].all()


def test_gate_needs_mixed_batches():
    from app.models.training.training_config import SpectralStemFinetuneConfig
    src = [{"name": "enmap", "train_dir": "t", "test_dir": "v", "share": 1.0}]
    assert SpectralStemFinetuneConfig(sources=src, gate_pct=1.0, mixed_batches=True).gate_pct == 1.0
    with pytest.raises(ValueError):
        SpectralStemFinetuneConfig(sources=src, gate_pct=1.0)


def test_block_inference_checkerboard_passes_are_complementary(tmp_path):
    from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
    from app.models.training.inference_config import InferenceConfig
    from app.models.training.training_config import SpectralStemFinetuneConfig
    wl = torch.linspace(420, 2440, 202)
    m = SpectralStemSegFormerMAE("reduced_attention", wl.tolist(), [0.2] * 202, [0.1] * 202, seg_classes=5,
                                 seg_decoder_dim=16, embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)
    cfg = SpectralStemFinetuneConfig(sources=[{"name": "e", "train_dir": "t", "test_dir": "v", "share": 1.0}],
                                     stem="reduced_attention", seg_classes=5, seg_decoder_dim=16, mask_block=2,
                                     embed_dims=[8, 16, 32, 64], num_heads=[1, 1, 2, 2], decoder_dim=32).model_dump()
    torch.save({"model_state_dict": m.state_dict(), "config": {"model_config": cfg}}, tmp_path / "m.pt")
    inf = SpectralStemInferencer(InferenceConfig(foundation_model_name="spectral_stem_finetune", model_config=cfg,
                                                 checkpoint_path=str(tmp_path / "m.pt"), patch_size=64, stride=32))
    valid = torch.ones(1, 1, 64, 64, device=inf.device)
    k1, k2 = (inf._checkerboard_keep_mask(64, 64, valid, inv) for inv in (False, True))
    p1, p2 = 1 - k1, 1 - k2
    assert torch.equal(p1 + p2, torch.ones_like(p1))                       # every token predicted exactly once
    g = p1.reshape(16, 16)
    assert torch.equal(g[0:2, 0:2], g[0, 0].expand(2, 2))                  # in 2 x 2 token cells
    assert g[0, 0] != g[0, 2]


def test_block_inference_never_predicts_a_pixel_from_a_pass_that_saw_it(tmp_path):
    """FT08 leak (fixed 2026-10-31): with mask_block 2, every pixel's prediction must come from the pass where its
    8 x 8 block was hidden - so changing the pixels inside one block cannot change that block's own predictions."""
    from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
    from app.models.training.inference_config import InferenceConfig
    from app.models.training.training_config import SpectralStemFinetuneConfig
    torch.manual_seed(0)
    wl = torch.linspace(420, 2440, 202)
    m = SpectralStemSegFormerMAE("reduced_attention", wl.tolist(), [0.2] * 202, [0.1] * 202, seg_classes=5,
                                 seg_decoder_dim=16, embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32)
    cfg = SpectralStemFinetuneConfig(sources=[{"name": "e", "train_dir": "t", "test_dir": "v", "share": 1.0}],
                                     stem="reduced_attention", seg_classes=5, seg_decoder_dim=16, mask_block=2,
                                     embed_dims=[8, 16, 32, 64], num_heads=[1, 1, 2, 2], decoder_dim=32).model_dump()
    torch.save({"model_state_dict": m.state_dict(), "config": {"model_config": cfg}}, tmp_path / "m.pt")
    inf = SpectralStemInferencer(InferenceConfig(foundation_model_name="spectral_stem_finetune", model_config=cfg,
                                                 checkpoint_path=str(tmp_path / "m.pt"), patch_size=64, stride=32))
    assert inf.config.checkerboard_cell_size == 2
    inf.set_wavelengths(wl)
    x = (torch.rand(1, 202, 64, 64) * 0.3).to(inf.device)
    valid = torch.ones(1, 1, 64, 64, device=inf.device)
    y = x.clone()
    y[:, :, 8:16, 16:24] += 0.4                                            # one 2 x 2-token (8 px) block
    with torch.no_grad():
        a, b = inf.predict(x, valid), inf.predict(y, valid)
    assert torch.allclose(a[..., 8:16, 16:24], b[..., 8:16, 16:24], atol=1e-5)
