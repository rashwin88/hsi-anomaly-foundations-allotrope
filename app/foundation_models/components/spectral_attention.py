"""
Cross-attention over a pixel's bands: the adaptive replacement for SpectralCompressor and
SpectralDecompressor.

Used by the next Indradhanu's spectral stem (variant C of the SpectralEarth pretraining
experiment, research/experiments/indradhanu-se-pretrain/); the SegFormer encoder and
decoder between the two are unchanged and still see (B, 32, H, W).

Why: the 1x1 compressor mixes every pixel with one fixed recipe, is tied to 165 band
positions, and cannot skip a dead band. Here 32 learned queries read each pixel's band
tokens - value plus wavelength (spectral_tokens.py) - so the mix depends on the pixel, any
band list works, and a dead band is masked to a weight of exactly 0. The decoder turns each
requested wavelength into a query against the pixel's 32 latent channels, so one set of
weights reconstructs any sensor's bands. Design: docs/proposals/spectral-attention-
bottleneck.md; taught in `indradhanu workings/31-*.md` to `38-*.md`.

Gotchas:
  - Everything runs per pixel: (B, N, H, W) is reshaped to B*H*W rows. Nothing moves
    between pixels, which is what keeps masked pixels from reconstructing themselves.
  - Memory, not parameters, is the cost: keys and values are (pixels, N, 32) - ~800 MiB per
    128x128 patch in fp32 (course part 36). Use bf16.
  - No BatchNorm inside; the model adds the BatchNorm2d all variants share.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.foundation_models.components.spectral_tokens import wavelength_encoding


# CUDA's fused attention kernels put the batch dimension on one launch-grid axis, capped at
# 65,535 blocks; with pixels as the batch, 4 patches of 128 x 128 (65,536 rows) already fail
# with "invalid configuration argument". Pixels are independent, so chunking changes nothing.
SDPA_MAX_ROWS = 32_768


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """scaled_dot_product_attention over (P, h, L, hd), at most SDPA_MAX_ROWS pixels per call."""
    if q.shape[0] <= SDPA_MAX_ROWS:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return torch.cat([
        F.scaled_dot_product_attention(q[i:i + SDPA_MAX_ROWS], k[i:i + SDPA_MAX_ROWS], v[i:i + SDPA_MAX_ROWS],
                                       attn_mask=None if mask is None else mask[i:i + SDPA_MAX_ROWS])
        for i in range(0, q.shape[0], SDPA_MAX_ROWS)])


def _per_pixel_wavelengths(wavelengths_nm: torch.Tensor, b: int, hw: int) -> torch.Tensor:
    """(N,) or (B, N) -> (N,) shared, or (B*H*W, N) one row per pixel."""
    if wavelengths_nm.dim() == 1:
        return wavelengths_nm
    return wavelengths_nm.repeat_interleave(hw, dim=0) if wavelengths_nm.shape[0] == b else wavelengths_nm


class SpectralCrossAttentionCompressor(nn.Module):
    """(B, N, H, W) normalised bands -> (B, D, H, W): D learned queries attend over each
    pixel's N band tokens. Parameters at D = d = 32: 5,345, whatever N is."""

    def __init__(self, compressed_channels: int = 32, token_dim: int = 32, heads: int = 4):
        super().__init__()
        if token_dim % heads:
            raise ValueError(f"token_dim {token_dim} must divide by heads {heads}")
        self.D, self.d, self.heads = compressed_channels, token_dim, heads
        self.value_embed = nn.Linear(1, token_dim)
        self.queries = nn.Parameter(torch.randn(compressed_channels, token_dim) * 0.02)
        self.q, self.k, self.v, self.o = (nn.Linear(token_dim, token_dim) for _ in range(4))
        self.to_channel = nn.Linear(token_dim, 1)

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor,
                band_valid: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, H, W). wavelengths_nm: (N,) or (B, N). band_valid: (B, N, H, W) bool,
        False = dead band, skipped exactly; None = all valid."""
        b, n, h, w = x.shape
        p, hd = b * h * w, self.d // self.heads
        rows = x.permute(0, 2, 3, 1).reshape(p, n, 1)                                   # (P, N, 1)
        wl = _per_pixel_wavelengths(wavelengths_nm.to(x.device), b, h * w)
        tokens = self.value_embed(rows) + wavelength_encoding(wl, self.d).to(x.dtype)  # (P, N, d)
        k = self.k(tokens).view(p, n, self.heads, hd).transpose(1, 2)                  # (P, h, N, hd)
        v = self.v(tokens).view(p, n, self.heads, hd).transpose(1, 2)
        q = self.q(self.queries).view(self.D, self.heads, hd).transpose(0, 1)          # (h, D, hd)
        q = q.unsqueeze(0).expand(p, -1, -1, -1)                                       # (P, h, D, hd)
        mask, empty = None, None
        if band_valid is not None:
            mask = band_valid.permute(0, 2, 3, 1).reshape(p, 1, 1, n)                  # True = attend
            # A pixel with no valid band would softmax over all -inf: NaN in a plain softmax,
            # zeros in torch 2.5's fused CPU kernel, unspecified elsewhere. Let it attend to
            # everything, then zero its output - invalid pixels are 0 throughout the codebase.
            empty = ~mask.any(dim=-1, keepdim=True)                                    # (P, 1, 1, 1)
            mask = mask | empty
        out = _attend(q, k, v, mask)                                                   # (P, h, D, hd)
        out = self.to_channel(self.o(out.transpose(1, 2).reshape(p, self.D, self.d)))  # (P, D, 1)
        if empty is not None:
            out = out.masked_fill(empty.view(p, 1, 1), 0.0)
        return out.view(b, h, w, self.D).permute(0, 3, 1, 2)                           # (B, D, H, W)



class SpectralCrossAttentionDecoder(nn.Module):
    """(B, D, H, W) latent channels + wavelengths -> (B, N, H, W) normalised bands: each
    requested wavelength is a query against the pixel's D channel tokens. Parameters at
    D = d = 32: 6,401, whatever N is (course part 37)."""

    def __init__(self, compressed_channels: int = 32, token_dim: int = 32, heads: int = 4):
        super().__init__()
        if token_dim % heads:
            raise ValueError(f"token_dim {token_dim} must divide by heads {heads}")
        self.D, self.d, self.heads = compressed_channels, token_dim, heads
        self.query_from_wavelength = nn.Linear(token_dim, token_dim)
        self.channel_value_embed = nn.Linear(1, token_dim)
        # Channels have no wavelength - channel 7 is whatever the compressor's 7th query
        # measured - so each gets a learned identity instead of a computed tag.
        self.channel_identity = nn.Parameter(torch.randn(compressed_channels, token_dim) * 0.02)
        self.q, self.k, self.v, self.o = (nn.Linear(token_dim, token_dim) for _ in range(4))
        self.to_value = nn.Linear(token_dim, 1)

    def forward(self, z: torch.Tensor, wavelengths_nm: torch.Tensor) -> torch.Tensor:
        """z: (B, D, H, W). wavelengths_nm: (N,) or (B, N), the bands to reconstruct."""
        b, _, h, w = z.shape
        p, hd = b * h * w, self.d // self.heads
        rows = z.permute(0, 2, 3, 1).reshape(p, self.D, 1)                                   # (P, D, 1)
        tokens = self.channel_value_embed(rows) + self.channel_identity                      # (P, D, d)
        k = self.k(tokens).view(p, self.D, self.heads, hd).transpose(1, 2)                   # (P, h, D, hd)
        v = self.v(tokens).view(p, self.D, self.heads, hd).transpose(1, 2)
        wl = _per_pixel_wavelengths(wavelengths_nm.to(z.device), b, h * w)
        n = wl.shape[-1]
        q = self.q(self.query_from_wavelength(wavelength_encoding(wl, self.d).to(z.dtype)))  # (N, d) or (P, N, d)
        q = q.view(-1, n, self.heads, hd).transpose(1, 2).expand(p, -1, -1, -1)              # (P, h, N, hd)
        out = _attend(q, k, v)                                                               # (P, h, N, hd)
        out = self.to_value(self.o(out.transpose(1, 2).reshape(p, n, self.d)))               # (P, N, 1)
        return out.view(b, h, w, n).permute(0, 3, 1, 2)                                      # (B, N, H, W)
