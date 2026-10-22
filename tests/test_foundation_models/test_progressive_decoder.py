"""FT09's progressive decoder: no hidden pixel leaks through the skip, FT04 warm-starts it, and it renders full
resolution (app/foundation_models/components/progressive_decoder.py)."""
import pytest
import torch

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE

WL = torch.linspace(420, 2440, 202)
STATS = (WL.tolist(), [0.2] * 202, [0.1] * 202)
SMALL = dict(embed_dims=(8, 16, 32, 64), num_heads=(1, 1, 2, 2), decoder_dim=32, seg_classes=5, seg_decoder_dim=16)


def model(decoder_type="progressive", seed=0):
    torch.manual_seed(seed)
    return SpectralStemSegFormerMAE("reduced_attention", *STATS, decoder_type=decoder_type, **SMALL).eval()


def test_hidden_pixels_never_reach_the_output():
    """Change only the pixels of hidden tokens: the whole reconstruction is unchanged."""
    m = model()
    x = torch.rand(2, 202, 64, 64) * 0.3
    keep = torch.ones(2, 256)
    keep[:, [0, 17, 100, 255]] = 0                         # four hidden 4 x 4 tokens per sample
    y = x.clone()
    for t in (0, 17, 100, 255):
        r, c = divmod(t, 16)
        y[:, :, 4 * r:4 * r + 4, 4 * c:4 * c + 4] = torch.rand(2, 202, 4, 4)
    with torch.no_grad():
        assert torch.allclose(m(x, WL, keep_mask=keep), m(y, WL, keep_mask=keep), atol=1e-6)
        groups = [(x, WL, None, keep, None)], [(y, WL, None, keep, None)]
        assert torch.allclose(m.forward_mixed(groups[0])[0], m.forward_mixed(groups[1])[0], atol=1e-6)


def test_visible_pixels_do_reach_the_output():
    """The skip is live: changing a visible pixel changes its own reconstruction."""
    m = model()
    x = torch.rand(1, 202, 64, 64) * 0.3
    y = x.clone()
    y[:, :, 30, 30] += 0.5
    with torch.no_grad():
        assert (m(x, WL) - m(y, WL))[0, :, 30, 30].abs().max() > 1e-4


def test_warm_start_from_a_segformer_decoder_checkpoint():
    old, new = model("segformer"), model("progressive", seed=1)
    state = {k: v for k, v in old.state_dict().items() if not k.startswith("decoder.refine.")}
    missing, unexpected = new.load_state_dict(state, strict=False)
    assert not unexpected and missing and all(k.startswith(("decoder.up_", "decoder.head.")) for k in missing)
    assert torch.equal(new.decoder.fuse_conv.weight, old.decoder.fuse_conv.weight)


def test_shapes_segmentation_and_bad_type():
    m = model()
    x = torch.rand(2, 101, 64, 64) * 0.3
    with torch.no_grad():
        assert m(x, WL[::2]).shape == (2, 101, 64, 64)
        assert m.segment(x, WL[::2]).shape == (2, 5, 64, 64)
    with pytest.raises(ValueError):
        model("unet")


def test_decoder_mode_is_a_valid_finetune_mode():
    from app.models.training.training_config import SpectralStemFinetuneConfig
    src = [{"name": "enmap", "train_dir": "t", "test_dir": "v", "share": 1.0}]
    c = SpectralStemFinetuneConfig(sources=src, mode="decoder", decoder_type="progressive", mixed_batches=True)
    assert (c.mode, c.decoder_type) == ("decoder", "progressive")
