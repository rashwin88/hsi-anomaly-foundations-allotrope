"""
The SegFormer MAE with a swappable spectral stem: the variants of the SpectralEarth
pretraining experiment in one class (A linear, B gate, C cross_attention, C' reduced_attention).

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
    reduced_attention SpectralReducedCompressor (C')         WavelengthLinearDecoder
    all              -> BatchNorm2d(D) -> SegFormer encoder/decoder (unchanged) ->

Gotchas:
  - Input and output are reflectance; normalisation by wavelength happens inside, from the
    curve in the checkpoint (WavelengthStats).
  - linear and gate are built for exactly `bands` inputs in one band order; the wavelength
    stems take any band list per call, and may reconstruct a different one (out_wavelengths_nm).
    Reading a list well still needs training on varied lists (docs/lld/spectral-stem.md).
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
                 band_weighting: str = "count", seg_classes: int = 0, seg_decoder_dim: int = 128,
                 embed_dims=(32, 64, 160, 256), num_heads=(1, 2, 5, 8), reduction_ratios=(8, 4, 2, 1),
                 num_blocks=(2, 2, 2, 2), decoder_dim: int = 256, expansion_ratio: int = 4,
                 drop_rate: float = 0.0, scale_decoder_dim: int = 0):
        super().__init__()
        if stem not in STEMS:
            raise ValueError(f"stem must be one of {STEMS}, got {stem!r}")
        self.stem, self.bands, D = stem, bands, compressed_channels
        self.stats = WavelengthStats(stats_wavelengths_nm, stats_mean, stats_std)
        if stem == "cross_attention":
            self.spectral_in = SpectralCrossAttentionCompressor(D, token_dim, attention_heads)
            self.spectral_out = SpectralCrossAttentionDecoder(D, token_dim, attention_heads)
        elif stem == "reduced_attention":
            self.spectral_in = SpectralReducedCompressor(D, token_dim, attention_heads, bin_nm, band_weighting)
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
        # Optional segmentation head on the same encoder features (fine-tuning stage 1+,
        # docs/proposals/fine-tuning-and-segmentation.md): one logit per class, independent (sigmoid),
        # predicted per pixel like the reconstruction. Absent (seg_classes = 0) in pretrained models.
        self.seg_head = (SegFormerDecoder(embed_dims=list(embed_dims), decoder_dim=seg_decoder_dim, out_channels=seg_classes)
                         if seg_classes else None)
        # Optional uncertainty head (docs/proposals/uncertainty-head.md): log b(p), the expected size of the
        # normalised reconstruction error at each pixel, predicted from the same (masked) encoder features - so a
        # hidden pixel's b comes from its context, never from the pixel itself. Absent (0) before FT06.
        self.scale_head = (SegFormerDecoder(embed_dims=list(embed_dims), decoder_dim=scale_decoder_dim, out_channels=1)
                           if scale_decoder_dim else None)
        # Every wavelength the model has trained on, sorted: starts as the pretraining list (the
        # normalisation curve) and grows as fine-tuning reads other sensors. The curve itself stays
        # frozen (docs/lld/spectral-stem.md); this record is what trained_band_mask reads.
        self.register_buffer("trained_wl", self.stats.wl.clone())

    def extend_trained_wavelengths(self, wavelengths_nm, tol_nm: float = 0.5) -> int:
        """Add wavelengths not within tol_nm of one already recorded; returns how many were added."""
        wl = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=self.trained_wl.device).flatten()
        new = torch.sort(wl[(wl[:, None] - self.trained_wl[None, :]).abs().min(dim=1).values > tol_nm]).values
        kept = []                                   # new wavelengths within tol_nm of each other count once
        for v in new.tolist():
            if not kept or v - kept[-1] > tol_nm:
                kept.append(v)
        if kept:
            self.trained_wl = torch.sort(torch.cat([self.trained_wl, torch.tensor(kept, device=self.trained_wl.device)])).values
        return len(kept)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from before 2026-10-16 have no record: fall back to their curve. Newer ones may hold
        # a longer record than this freshly built model, so take its size from the checkpoint.
        key, curve = prefix + "trained_wl", state_dict.get(prefix + "stats.wl")
        if key not in state_dict and curve is not None:
            state_dict[key] = torch.sort(curve).values
        if key in state_dict:
            self.trained_wl = torch.empty_like(state_dict[key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def trained_band_mask(self, wavelengths_nm, tol_nm: float = 15.0) -> torch.Tensor:
        """(N,) bool: True where a band lies within tol_nm of a wavelength the model has trained on
        (trained_wl). Bands outside - another sensor's water-vapour bands, or beyond everything trained -
        meet untrained wavelength networks, so callers drop them."""
        wl = torch.as_tensor(wavelengths_nm, dtype=torch.float32, device=self.trained_wl.device)
        return (wl[:, None] - self.trained_wl[None, :]).abs().min(dim=1).values <= tol_nm

    def encode(self, x: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None = None,
               keep_mask: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Reflectance (B, N, H, W) -> the encoder's four stage features (MAE token removal if keep_mask)."""
        if self.stem not in WAVELENGTH_STEMS and x.shape[1] != self.bands:
            raise ValueError(f"{self.stem} stem was built for {self.bands} bands, got {x.shape[1]}")
        return self.encoder(self.norm(self._stem(x, wavelengths_nm, band_valid)), keep_mask=keep_mask)

    def _stem(self, x: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None) -> torch.Tensor:
        """(B, N, H, W) reflectance -> (B, D, H, W): normalise, zero dead voxels, spectral stem."""
        z = self.stats.normalize(x, wavelengths_nm)
        if band_valid is not None:
            z = z.masked_fill(~band_valid, 0.0)
        return self.spectral_in(z, wavelengths_nm, band_valid) if self.stem in WAVELENGTH_STEMS else self.spectral_in(z)

    def segment_mixed(self, groups: list) -> list:
        """segment() over samples with different band lists: groups [(x, wavelengths_nm, band_valid)]. One stem pass
        per group, then one unmasked encoder pass and the head over the concatenated batch (BatchNorm sees every
        group). Returns each group's (B_g, seg_classes, H, W) logits."""
        if self.seg_head is None:
            raise ValueError("this model has no segmentation head (seg_classes = 0)")
        if self.stem not in WAVELENGTH_STEMS:
            raise ValueError(f"{self.stem} stem reads one band list only")
        z = torch.cat([self._stem(x, wl, bv) for x, wl, bv in groups])
        logits = self.seg_head(self.encoder(self.norm(z)))
        return list(torch.split(logits, [len(x) for x, _, _ in groups]))

    def forward_mixed(self, groups: list, with_scale: bool = False):
        """One pass over samples with different band lists. groups: [(x, wavelengths_nm, band_valid, keep_mask,
        out_wavelengths_nm)], each as in forward (keep_mask set for all groups or none; out may be None). Each group
        has its own stem pass; the stem outputs share one batch through BatchNorm, encoder and decoder, so batch
        statistics and gradients mix sensors. Returns each group's (B_g, M_g, H, W) reconstruction; with_scale, also
        each group's (B_g, 1, H, W) log b from the uncertainty head."""
        if self.stem not in WAVELENGTH_STEMS:
            raise ValueError(f"{self.stem} stem reads one band list only")
        z = torch.cat([self._stem(x, wl, bv) for x, wl, bv, _, _ in groups])
        keep = None if groups[0][3] is None else torch.cat([g[3] for g in groups])
        features = self.encoder(self.norm(z), keep_mask=keep)
        z_hat = self.decoder(features)
        out, start = [], 0
        for x, wl, _, _, out_wl in groups:
            wl_out = wl if out_wl is None else out_wl
            part = self.spectral_out(z_hat[start:start + len(x)], wl_out)
            out.append(self.stats.denormalize(part, wl_out))
            start += len(x)
        if with_scale:
            return out, list(torch.split(self.scale_head(features), [len(g[0]) for g in groups]))
        return out

    def segment(self, x: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None = None) -> torch.Tensor:
        """(B, N, H, W) reflectance -> (B, seg_classes, H, W) logits, from the unmasked encoder."""
        if self.seg_head is None:
            raise ValueError("this model has no segmentation head (seg_classes = 0)")
        return self.seg_head(self.encode(x, wavelengths_nm, band_valid))

    def forward(self, x: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None = None,
                keep_mask: torch.Tensor | None = None, out_wavelengths_nm: torch.Tensor | None = None,
                with_scale: bool = False):
        """x: (B, N, H, W) reflectance. wavelengths_nm: (N,) or (B, N). band_valid: (B, N, H, W)
        bool, False = dead voxel. keep_mask: (B, tokens) MAE keep mask for the encoder.
        out_wavelengths_nm: (M,) bands to reconstruct, default the input's - wavelength stems
        only (band-list augmentation reads a subset and reconstructs every band).
        Returns (B, M, H, W) reconstructed reflectance (with_scale: and the (B, 1, H, W) log b). Raises ValueError if a fixed-band
        stem gets a band count other than `bands`, or out_wavelengths_nm."""
        if out_wavelengths_nm is not None and self.stem not in WAVELENGTH_STEMS:
            raise ValueError(f"{self.stem} stem reconstructs its input bands only")
        out_wl = wavelengths_nm if out_wavelengths_nm is None else out_wavelengths_nm
        features = self.encode(x, wavelengths_nm, band_valid, keep_mask)
        z_hat = self.decoder(features)                                    # (B, D, H, W)
        if self.stem in WAVELENGTH_STEMS:
            z_hat = self.spectral_out(z_hat, out_wl)                         # (B, M, H, W)
        else:
            z_hat = self.spectral_out(z_hat)
        recon = self.stats.denormalize(z_hat, out_wl)
        return (recon, self.scale_head(features)) if with_scale else recon
