"""
Spectral reduction: the cross-attention stem made cheap without averaging away detail.

Used as the `reduced_attention` stem of SpectralStemSegFormerMAE - the successor to the
per-band cross-attention stem (spectral_attention.py) for the SpectralEarth pretraining runs.

Why: the per-band stem builds a (pixels x bands x 32) token for every band of every pixel and
keeps ~19 such tensors for the backward pass - 3.7 GiB per 128 x 128 patch, ~5 patches/s on a
20 GB GPU (S01). SegFormer's efficient attention shrinks keys and values before attending; this
does the same along wavelength: ~42 bin tokens instead of 202 band tokens. Three things keep the
detail a plain bin average would lose:
  1. Every band gets its own weight vector inside its bin: token_g = mean_b(v_b * k(lambda_b)).
     A bin holds ~5 bands mapped into 32 numbers, so the map can be one-to-one - a narrow dip
     and its shoulder stay distinguishable. (A shared weight, as in the per-band stem's value
     embedding, would keep only the bin's mean.)
  2. A per-band skip path straight to the 32 channels: mean_b(v_b * W(lambda_b)) - today's 1x1
     compressor with weights computed from wavelength, so it works for any band list. Attention
     over bins adds adaptivity on top; it does not carry all the detail alone.
  3. The decoder is per band: out_n = U(lambda_n) . z + c(lambda_n), weights computed from each
     requested wavelength - today's decompressor generalised to any band list, no attention.
k, W, U are small networks of the wavelength encoding (spectral_tokens.py), so no parameter is
tied to a band position. Dead bands drop out of every mean exactly (masked before summing).

Gotchas:
  - Bins are a fixed grid from 400 nm, so bin g means the same wavelengths on every sensor.
  - Everything is matrix products over the band axis: nothing (pixels x bands x 32) is stored.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.foundation_models.components.spectral_attention import _attend
from app.foundation_models.components.spectral_tokens import wavelength_encoding

BIN_ORIGIN_NM, BIN_END_NM = 400.0, 2500.0
# A band beside an excluded region (water vapour) would otherwise claim the whole gap.
MAX_GAP_NM = 20.0


def band_widths(wavelengths_nm: torch.Tensor, max_gap_nm: float = MAX_GAP_NM) -> torch.Tensor:
    """(N,) wavelengths in any order -> (N,) nm of spectrum each band stands for: half the gap to
    each neighbour (the one gap at either end), gaps capped at max_gap_nm. Weighting averages by
    this makes them approximate integrals over wavelength, whatever the sensor's band spacing."""
    wl = wavelengths_nm.float()
    if wl.numel() < 2:
        return torch.ones_like(wl)
    order = torch.argsort(wl)
    gaps = torch.diff(wl[order]).clamp(max=max_gap_nm)
    sorted_w = torch.cat([gaps[:1], (gaps[:-1] + gaps[1:]) / 2, gaps[-1:]])
    return torch.empty_like(wl).scatter_(0, order, sorted_w)


def _hyper(out_dim: int, enc_dim: int = 32, hidden: int = 64) -> nn.Module:
    """Wavelength encoding -> a weight vector: the small network behind k, W and U."""
    return nn.Sequential(nn.Linear(enc_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def _rows(x: torch.Tensor) -> torch.Tensor:
    b, n, h, w = x.shape
    return x.permute(0, 2, 3, 1).reshape(b * h * w, n)


class SpectralReducedCompressor(nn.Module):
    """(B, N, H, W) normalised bands + wavelengths (N,) -> (B, D, H, W)."""

    def __init__(self, compressed_channels: int = 32, token_dim: int = 32, heads: int = 4, bin_nm: float = 40.0,
                 band_weighting: str = "count"):
        super().__init__()
        if token_dim % heads:
            raise ValueError(f"token_dim {token_dim} must divide by heads {heads}")
        if band_weighting not in ("count", "width"):
            raise ValueError(f"band_weighting must be 'count' or 'width', got {band_weighting!r}")
        self.D, self.d, self.heads, self.bin_nm = compressed_channels, token_dim, heads, bin_nm
        self.band_weighting = band_weighting
        self.n_bins = int(math.ceil((BIN_END_NM - BIN_ORIGIN_NM) / bin_nm))
        self.band_weight = _hyper(token_dim)              # fix 1: k(lambda), per band inside a bin
        self.skip_weight = _hyper(compressed_channels)    # fix 2: W(lambda), per-band skip path
        self.bin_bias = nn.Linear(token_dim, token_dim)   # where the bin sits (its centre's encoding)
        self.queries = nn.Parameter(torch.randn(compressed_channels, token_dim) * 0.02)
        self.q, self.k, self.v, self.o = (nn.Linear(token_dim, token_dim) for _ in range(4))
        self.to_channel = nn.Linear(token_dim, 1)

    def bin_index(self, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        return ((wavelengths_nm - BIN_ORIGIN_NM) / self.bin_nm).floor().long().clamp(0, self.n_bins - 1)

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor,
                band_valid: torch.Tensor | None = None) -> torch.Tensor:
        if wavelengths_nm.dim() != 1:
            raise ValueError("reduced stem takes one band list per batch: wavelengths (N,)")
        b, n, h, w = x.shape
        p, hd, G = b * h * w, self.d // self.heads, self.n_bins
        wl = wavelengths_nm.to(x.device).float()
        enc = wavelength_encoding(wl, self.d).to(x.dtype)                           # (N, d)
        valid = torch.ones_like(x) if band_valid is None else band_valid.to(x.dtype)
        if self.band_weighting == "width":                  # each band counts by the nm it covers
            valid = valid * band_widths(wl).to(x.dtype).view(1, n, 1, 1)
        vals = _rows(x * valid)                                                      # (P, N), dead -> 0
        cnt_band = _rows(valid)                                                      # (P, N) weight per band
        onehot = F.one_hot(self.bin_index(wl), G).to(x.dtype)                        # (N, G)
        cnt = cnt_band @ onehot                                                      # (P, G) valid bands per bin
        # fix 1: token_g = mean over the bin's valid bands of v_b * k(lambda_b) - one matmul, no (P, N, d)
        kmat = (onehot[:, :, None] * self.band_weight(enc)[:, None, :]).reshape(n, G * self.d)
        tok = (vals @ kmat).view(p, G, self.d) / cnt.clamp(min=1)[..., None]
        centres = BIN_ORIGIN_NM + self.bin_nm * (torch.arange(G, device=x.device, dtype=torch.float32) + 0.5)
        tok = tok + self.bin_bias(wavelength_encoding(centres, self.d).to(x.dtype))[None]
        k = self.k(tok).view(p, G, self.heads, hd).transpose(1, 2)                  # (P, h, G, hd)
        v = self.v(tok).view(p, G, self.heads, hd).transpose(1, 2)
        q = self.q(self.queries).view(self.D, self.heads, hd).transpose(0, 1).unsqueeze(0).expand(p, -1, -1, -1)
        mask = (cnt > 0).view(p, 1, 1, G)
        empty = ~mask.any(dim=-1, keepdim=True)                                      # pixel with no valid band
        out = _attend(q, k, v, mask | empty)
        out = self.to_channel(self.o(out.transpose(1, 2).reshape(p, self.D, self.d))).squeeze(-1)   # (P, D)
        # fix 2: the per-band skip path - every valid band reaches the channels with its own weights
        skip = (vals @ self.skip_weight(enc)) / cnt_band.sum(1, keepdim=True).clamp(min=1)          # (P, D)
        z = (out + skip).masked_fill(empty.view(p, 1), 0.0)
        return z.view(b, h, w, self.D).permute(0, 3, 1, 2)


class WavelengthLinearDecoder(nn.Module):
    """fix 3: (B, D, H, W) latent + wavelengths (N,) -> (B, N, H, W), one output weight vector per
    requested wavelength: out_n = U(lambda_n) . z + c(lambda_n)."""

    def __init__(self, compressed_channels: int = 32, enc_dim: int = 32):
        super().__init__()
        self.D, self.enc_dim = compressed_channels, enc_dim
        self.out_weight = _hyper(compressed_channels + 1, enc_dim)

    def forward(self, z: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        if wavelengths_nm.dim() != 1:
            raise ValueError("reduced stem takes one band list per batch: wavelengths (N,)")
        uc = self.out_weight(wavelength_encoding(wavelengths_nm.to(z.device).float(), self.enc_dim).to(z.dtype))
        u, c = uc[:, : self.D], uc[:, self.D]                                        # (N, D), (N,)
        return torch.einsum("bdhw,nd->bnhw", z, u) + c.view(1, -1, 1, 1)
