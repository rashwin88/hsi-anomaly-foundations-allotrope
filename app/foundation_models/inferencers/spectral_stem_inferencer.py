"""
Full-scene inference for spectral-stem checkpoints (SpectralStemSegFormerMAE: linear / gate /
cross_attention / reduced_attention), with exactly the product's patching and masking.

Used by the product's anomaly_scoring (Indradhanu V2) and by app/evaluation/models.py (scripts/evaluate.py) -
one implementation of band reading, reconstruction and segmentation for both.

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

import numpy as np
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

    def read_bands(self, cube: np.ndarray, wavelengths_nm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(cube, wavelengths, kept-band index) with only the bands the model has trained on (trained_band_mask,
        docs/lld/spectral-stem.md); also sets the wavelengths for the next predict_full_scene / segment_full_scene."""
        wl = np.asarray(wavelengths_nm, dtype=np.float64)
        keep = self.model.inner.trained_band_mask(wl).cpu().numpy()
        self.set_wavelengths(wl[keep])
        return cube[keep], wl[keep], np.flatnonzero(keep)

    SEG_TILE, SEG_STRIDE, SEG_BATCH = 128, 64, 16
    SEG_CLASSES = ("water", "cloud", "thin_cloud", "shadow", "snow")    # the head's output order
    CLASS_NAMES = ("land", "water", "cloud", "shadow", "snow")          # segment_full_scene's class ids

    def segment_full_scene(self, cube: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(class map (H, W) int64: 0 land, 1 water, 2 cloud, 3 shadow, 4 snow; probabilities (5, H, W) float32).
        128 px windows at stride 64, sigmoid probabilities averaged where windows overlap; classes by priority
        cloud (cloud or thin cloud) > snow > shadow > water > land; 0 outside valid. Call read_bands first."""
        net = self.model.inner.eval()
        if net.seg_head is None:
            raise ValueError("checkpoint has no segmentation head")
        T, S = self.SEG_TILE, self.SEG_STRIDE
        C, H, W = cube.shape
        pad = [(-(n - T) % S) if n > T else T - n for n in (H, W)]
        x = np.pad(cube, ((0, 0), (0, pad[0]), (0, pad[1])))
        acc, cnt = np.zeros((5,) + x.shape[1:], np.float32), np.zeros(x.shape[1:], np.float32)
        pos = [(y, z) for y in range(0, x.shape[1] - T + 1, S) for z in range(0, x.shape[2] - T + 1, S)]
        for i in range(0, len(pos), self.SEG_BATCH):
            chunk = pos[i:i + self.SEG_BATCH]
            xb = torch.from_numpy(np.stack([x[:, y:y + T, z:z + T] for y, z in chunk])).to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                p = torch.sigmoid(net.segment(xb, self.model.wavelengths_nm).float()).cpu().numpy()
            for (y, z), pp in zip(chunk, p):
                acc[:, y:y + T, z:z + T] += pp
                cnt[y:y + T, z:z + T] += 1
        prob = (acc / np.maximum(cnt, 1))[:, :H, :W]
        cls = np.zeros((H, W), np.int64)
        cls[prob[0] > 0.5] = 1
        cls[prob[3] > 0.5] = 3
        cls[prob[4] > 0.5] = 4
        cls[np.maximum(prob[1], prob[2]) > 0.5] = 2
        return cls * valid.astype(bool), prob

