"""
Gated per-scene adaptation: a few steps of masked reconstruction on the scene's own tiles, starting from the shipped
weights, with the pixels that already look anomalous - and a halo around them - left out of the loss.

Why (research/experiments/scene-adaptation/): residual Mahalanobis absorbs model gains, and every attempt to learn
"normal" from a scene's context learned its targets too - they sit in a few small built-up surfaces. GT-HAD (reproduced
on the same PRISMA scenes, research/experiments/gt-had-prisma/) trains on the scene itself and was best after 5
epochs, not 150: a network learns the common background before the rare targets. Its gating keeps likely anomalies
out of what the network learns from. Here: start from the pretrained model (not from scratch), adapt briefly, and
gate on the current anomaly score.

    gate  = top gate_pct % of the scene's current score over valid pixels, dilated by halo px (the PSF halo and
            target parts in neighbouring tokens are what let FT06's head "see" a hidden target)
    loss  = L1 + sam_weight x SAM on hidden, valid, un-gated pixels - the fine-tuning loss (docs/lld/finetune.md)

The model runs in eval mode (stored BatchNorm statistics, no dropout); only the reconstruction path moves - the
segmentation head is outside the loss. The caller restores the original weights between scenes.
"""

from __future__ import annotations

import numpy as np
import torch

from app.foundation_models.components.sam_loss import SAMLoss
from app.foundation_models.components.token_masking import TokenMasking

TOKEN = 4                     # stage-1 patch embedding: kernel 4, stride 4, padding 0
MIN_TILE_VALID = 0.4          # tiles with less valid area are not drawn (the training shards' rule)


def gate_mask(score: np.ndarray, valid: np.ndarray, pct: float, halo: int) -> np.ndarray:
    """(H, W) bool, True = left out of the adaptation loss: the top pct % of score over valid pixels, dilated by
    halo px. pct = 0 gates nothing."""
    from scipy import ndimage
    if pct <= 0:
        return np.zeros(valid.shape, bool)
    gate = valid & (score >= np.percentile(score[valid], 100 - pct))
    return ndimage.binary_dilation(gate, iterations=halo) if halo else gate


def adapt(model, cube: np.ndarray, wavelengths_nm, valid: np.ndarray, gate: np.ndarray, steps: int,
          lr: float = 1e-5, tile: int = 128, batch: int = 8, mask_ratio: float = 0.65, sam_weight: float = 0.5,
          seed: int = 0, device=None) -> list[float]:
    """Adapt SpectralStemSegFormerMAE `model` in place to one scene; returns the per-step losses.
    cube (C, H, W) reflectance at wavelengths_nm, valid (H, W) bool, gate (H, W) bool from gate_mask."""
    device = device or next(model.parameters()).device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)                     # generate_prediction_mask draws from torch's global generator
    C, H, W = cube.shape
    if H < tile or W < tile:
        raise ValueError(f"scene {H}x{W} is smaller than one {tile} px tile")
    frac = _valid_fraction(valid, tile)
    ys, xs = np.nonzero(frac >= MIN_TILE_VALID)
    if not len(ys):
        return []
    wl = torch.as_tensor(np.asarray(wavelengths_nm), dtype=torch.float32, device=device)
    learn = valid & ~gate
    params = [p for n, p in model.named_parameters() if not n.startswith(("seg_head.", "scale_head."))]
    opt = torch.optim.Adam(params, lr=lr)
    sam = SAMLoss()
    model.eval()
    losses = []
    for _ in range(steps):
        pick = rng.integers(len(ys), size=batch)
        x = torch.stack([torch.from_numpy(cube[:, y:y + tile, z:z + tile]) for y, z in zip(ys[pick], xs[pick])]).to(device)
        v = torch.stack([torch.from_numpy(valid[y:y + tile, z:z + tile]) for y, z in zip(ys[pick], xs[pick])]).to(device)
        l = torch.stack([torch.from_numpy(learn[y:y + tile, z:z + tile]) for y, z in zip(ys[pick], xs[pick])]).to(device)
        pixel = v[:, None].float()
        tokens = TokenMasking.pixel_mask_to_token_mask(pixel, kernel_size=TOKEN, stride=TOKEN, padding=0)
        keep, pred = TokenMasking.generate_prediction_mask(tokens, mask_ratio, device)
        hidden = torch.nn.functional.interpolate(pred.reshape(batch, 1, tile // TOKEN, tile // TOKEN), size=(tile, tile),
                                                 mode="nearest")
        loss_mask = hidden * l[:, None].float()
        if loss_mask.sum() == 0:
            continue
        x_hat = model(x, wl, v[:, None].expand(-1, C, -1, -1), keep)
        l1 = ((x_hat - x).abs().mean(1, keepdim=True) * loss_mask).sum() / loss_mask.sum()
        loss = l1 + sam_weight * sam(x_hat, x, loss_mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    model.eval()
    return losses


def _valid_fraction(valid: np.ndarray, tile: int) -> np.ndarray:
    """Valid share of the tile whose top-left corner is at each (y, x) that fits a whole tile."""
    I = np.pad(valid.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = I[tile:, tile:] - I[:-tile, tile:] - I[tile:, :-tile] + I[:-tile, :-tile]
    return s / tile ** 2
