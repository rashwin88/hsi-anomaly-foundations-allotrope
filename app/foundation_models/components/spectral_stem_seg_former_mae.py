"""
The SegFormer MAE with a swappable spectral stem: the three variants of the SpectralEarth
pretraining experiment in one class.

Built by SpectralStemMAETrainer (app/foundation_models/trainers/spectral_stem_mae_trainer.py)
from a SpectralStemMAEConfig. HyperspectralSegFormerMAE - today's Indradhanu - is left as it
is, so its checkpoints keep loading.

Why one class: the experiment must change the spectral stem and nothing else. Normalisation,
the BatchNorm in front of the encoder, the SegFormer encoder and decoder are shared code, so a
difference between variants can only come from the stem.

    stem             in                                      out
    linear  (A)      Conv2d(N, D, 1)                         Conv2d(D, N, 1)
    gate    (B)      SpectralGate -> Conv2d(N, D, 1)         Conv2d(D, N, 1)
    cross_attention  SpectralCrossAttentionCompressor (C)    SpectralCrossAttentionDecoder
    all              -> BatchNorm2d(D) -> SegFormer encoder/decoder (unchanged) ->

Gotchas:
  - Input and output are reflectance; normalisation by wavelength happens inside, from the
    curve in the checkpoint (WavelengthStats).
  - linear and gate are built for exactly `bands` inputs in one band order; cross_attention
    takes any band list per call.
  - Dead voxels are set to 0 after normalisation (the curve's mean) for linear and gate,
    which cannot skip them; cross_attention masks them out exactly.
"""

import torch
import torch.nn as nn

from app.foundation_models.components.pixel_normalization import WavelengthStats
from app.foundation_models.components.seg_former_decoder import SegFormerDecoder
from app.foundation_models.components.seg_former_encoder import SegFormerEncoder
from app.foundation_models.components.spectral_attention import (
    SpectralCrossAttentionCompressor, SpectralCrossAttentionDecoder)
from app.foundation_models.components.spectral_gate import SpectralGate
from app.foundation_models.components.spectral_reduced import (
    SpectralReducedCompressor, WavelengthLinearDecoder)

STEMS = ("linear", "gate", "cross_attention", "reduced_attention")
# Stems that take any band list per call (wavelengths in, wavelengths out).
WAVELENGTH_STEMS = ("cross_attention", "reduced_attention")


class SpectralStemSegFormerMAE(nn.Module):
    def __init__(self, stem: str, stats_wavelengths_nm: list[float], stats_mean: list[float],
                 stats_std: list[float], bands: int = 202, compressed_channels: int = 32,
                 token_dim: int = 32, attention_heads: int = 4, gate_hidden: int = 32, bin_nm: float = 40.0,
                 embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8), reduction_ratios=(8, 4, 2, 1),
                 num_blocks=(2, 2, 2, 2), decoder_dim: int = 256, expansion_ratio: int = 4,
                 drop_rate: float = 0.0):
        super().__init__()
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}, got {stem!r}")
        self.stem, self.bands, D = stem, bands, compressed_channels
        self.stats = WavelengthStats(stats_wavelengths_nm, stats_mean, stats_std)
        if stem == "cross_attention":
            self.spectral_in = SpectralCrossAttentionCompressor(D, token_dim, attention_heads)
            self.spectral_out = SpectralCrossAttentionDecoder(D, token_dim, attention_heads)
        elif stem == "reduced_attention":
            self.spectral_in = SpectralReducedCompressor(D, token_dim, attention_heads, bin_nm)
            self.spectral_out = WavelengthLinearDecoder(D, token_dim)
        else:
            mix_in = nn.Conv2d(bands, D, kernel_size=1)
            self.spectral_in = nn.Sequential(SpectralGate(bands, gate_hidden), mix_in) if stem == "gate" else mix_in
            self.spectral_out = nn.Conv2d(D, bands, kernel_size=1)
        # Shared by all three stems (design decision 8): the encoder sees the same
        # distribution whichever stem produced it.
        self.norm = nn.BatchNorm2d(D)
        self.encoder = SegFormerEncoder(
            in_channels=D, embed_dims=list(embed_dims), num_heads=list(num_heads),
            reduction_ratios=list(reduction_ratios), num_blocks=list(num_blocks),
            expansion_ratio=expansion_ratio, drop_rate=drop_rate)
        self.decoder = SegFormerDecoder(embed_dims=list(embed_dims), decoder_dim=decoder_dim, out_channels=D)

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None = None,
                keep_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, H, W) reflectance. wavelengths_nm: (N,) or (B, N). band_valid: (B, N, H, W)
        bool, False = dead voxel. keep_mask: (B, tokens) MAE keep mask for the encoder.
        Returns (B, N, H, W) reconstructed reflectance. Raises ValueError if a fixed-band
        stem gets a band count other than `bands`."""
        if self.stem not in WAVELENGTH_STEMS and x.shape[1] != self.bands:
            raise ValueError(f"{self.stem} stem was built for {self.bands} bands, got {x.shape[1]}")
        z = self.stats.normalize(x, wavelengths_nm)                       # (B, N, H, W) z-scores
        if band_valid is not None:
            z = z.masked_fill(~band_valid, 0.0)
        if self.stem in WAVELENGTH_STEMS:
            z = self.spectral_in(z, wavelengths_nm, band_valid)              # (B, D, H, W)
        else:
            z = self.spectral_in(z)
        features = self.encoder(self.norm(z), keep_mask=keep_mask)
        z_hat = self.decoder(features)                                    # (B, D, H, W)
        if self.stem in WAVELENGTH_STEMS:
            z_hat = self.spectral_out(z_hat, wavelengths_nm)                 # (B, N, H, W)
        else:
            z_hat = self.spectral_out(z_hat)
        return self.stats.denormalize(z_hat, wavelengths_nm)
