"""
Full-scene inference for spectral-stem checkpoints (SpectralStemSegFormerMAE: linear / gate /
cross_attention / reduced_attention), with exactly the product's patching and masking.

Used by app/evaluation/models.py (scripts/evaluate.py); not yet wired into the product's anomaly_scoring.

Why a subclass of the product's SegFormer inferencer: a detection comparison is only fair if every
model is tiled, masked and stitched the same way. SegFormerMAEInferencer owns all of that - sliding
128 px patches, two complementary token masks so every pixel is predicted while hidden, the
erosion and overlap averaging - and calls `self.model(patch, mask=..., keep_mask=...)`. The new
models take wavelengths and per-band validity instead, so a thin adapter turns that call into
theirs; nothing else changes.

Gotchas:
  - Call `set_wavelengths(...)` with the scene's band centres before `predict_full_scene`.
  - The normalisation curve and the architecture come from the checkpoint itself (its
    `stats.*` buffers and saved config), so `pixel_stats_path` is not needed.
"""

import torch
import torch.nn as nn

from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.inferencers.segformer_mae_inferencer import SegFormerMAEInferencer


class _ProductCallAdapter(nn.Module):
    """model(x, mask=, keep_mask=) -> spectral-stem model(x, wavelengths, band_valid, keep_mask)."""

    def __init__(self, inner: SpectralStemSegFormerMAE):
        super().__init__()
        self.inner = inner
        self.wavelengths_nm: torch.Tensor | None = None

    def forward(self, x, mask=None, keep_mask=None):
        if self.wavelengths_nm is None:
            raise RuntimeError("set_wavelengths() before inference")
        band_valid = None if mask is None else mask.bool().expand(-1, x.shape[1], -1, -1)
        return self.inner(x, self.wavelengths_nm, band_valid, keep_mask)


class SpectralStemInferencer(SegFormerMAEInferencer):

    def build_model(self) -> nn.Module:
        cfg = self.config.model_config_
        sd = torch.load(self.config.checkpoint_path, map_location="cpu", weights_only=False)["model_state_dict"]
        inner = SpectralStemSegFormerMAE(
            stem=cfg.stem, stats_wavelengths_nm=sd["stats.wl"].tolist(), stats_mean=sd["stats.mean"].tolist(),
            stats_std=sd["stats.std"].tolist(), bands=cfg.bands, compressed_channels=cfg.compressed_channels,
            token_dim=cfg.token_dim, attention_heads=cfg.attention_heads, gate_hidden=cfg.gate_hidden,
            bin_nm=cfg.bin_nm, band_weighting=cfg.band_weighting,
            # fine-tuned checkpoints carry a segmentation head (spectral_stem_finetune); pretrained ones do not
            seg_classes=getattr(cfg, "seg_classes", 0), seg_decoder_dim=getattr(cfg, "seg_decoder_dim", 128),
            embed_dims=cfg.embed_dims, num_heads=cfg.num_heads,
            reduction_ratios=cfg.reduction_ratios, num_blocks=cfg.num_blocks, decoder_dim=cfg.decoder_dim,
            expansion_ratio=cfg.expansion_ratio, drop_rate=0.0)
        return _ProductCallAdapter(inner)

    def _load_weights(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.inner.load_state_dict(ckpt["model_state_dict"])

    def set_wavelengths(self, wavelengths_nm) -> None:
        """Band centres (nm) of the scene about to be predicted, in its channel order."""
        self.model.wavelengths_nm = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=self.device)
