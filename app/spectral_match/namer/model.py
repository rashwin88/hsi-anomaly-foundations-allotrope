"""
SpectralNamer: a dual encoder on the wavelength-aware stem (docs/proposals/spectral-namer.md).

    pixel side:   x, r (ring mean) -> stem -> [z_x, z_r, z_x - z_r] -> MLP -> h
                  h -> embedding (unit, 128) | fill in [0, 1] | (level-1 linear head, unused with level1="matching")
    library side: t -> stem -> MLP -> embedding (unit, 128)
    class score:  logsumexp over the class's library embeddings of  cos(z, e_j) / tau
    level 1:      "matching" (N001b on) - natural entries as one more class, level1_logit; or the linear head (N001)

Inputs are reflectance at any band list; the stem normalises by wavelength (WavelengthStats, Indradhanu Heavy's
curve - never recomputed) and reads any number of bands. The library embeddings depend only on the spectra, so a new
library is encoded, not trained.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.foundation_models.components.pixel_normalization import WavelengthStats
from app.foundation_models.components.spectral_reduced import SpectralReducedCompressor


def _mlp(i: int, h: int, o: int, dropout: float = 0.0) -> nn.Sequential:
    if not dropout:                                   # the N001/N001b layout: checkpoints keep loading
        return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, o))
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(h, o))


class SpectralNamer(nn.Module):
    def __init__(self, stats_wavelengths_nm, stats_mean, stats_std, compressed_channels: int = 64,
                 token_dim: int = 32, attention_heads: int = 4, bin_nm: float = 40.0, band_weighting: str = "width",
                 hidden: int = 256, embed: int = 128, dropout: float = 0.0):
        super().__init__()
        D = compressed_channels
        self.stats = WavelengthStats(stats_wavelengths_nm, stats_mean, stats_std)
        self.stem = SpectralReducedCompressor(D, token_dim, attention_heads, bin_nm, band_weighting)
        self.norm = nn.BatchNorm1d(D)                    # Heavy's post-stem BatchNorm, per pixel
        self.pixel = _mlp(3 * D, hidden, hidden, dropout)
        self.to_embed, self.to_manmade, self.to_fill = nn.Linear(hidden, embed), nn.Linear(hidden, 1), nn.Linear(hidden, 1)
        self.library = _mlp(D, hidden, embed, dropout)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))
        self.l1_bias = nn.Parameter(torch.zeros(()))      # level 1 by matching (level1_logit)

    def level1_logit(self, scores: torch.Tensor) -> torch.Tensor:
        """Level 1 by matching: scores (P, C + 1) from class_scores with natural entries as class C ->
        logsumexp over the man-made classes - the natural score + a learned bias (N001 showed a bare linear head on h
        learning natural vs man-made far slower than the matched classes)."""
        return torch.logsumexp(scores[:, :-1], dim=1) - scores[:, -1] + self.l1_bias

    def train(self, mode: bool = True):
        """The post-stem BatchNorm stays in eval mode: V2's running statistics (real pixels of the three sensors)
        normalise pixels, ring means and library spectra alike. In train mode each of the three encode calls would be
        normalised by its own batch, and evaluation by a running blend of all three - a train / eval mismatch."""
        super().train(mode)
        self.norm.eval()
        return self

    def encode(self, spectra: torch.Tensor, wavelengths_nm: torch.Tensor, band_valid: torch.Tensor | None = None):
        """(P, N) reflectance -> (P, D) stem features."""
        x = spectra[:, :, None, None]
        z = self.stats.normalize(x, wavelengths_nm)
        bv = None if band_valid is None else band_valid[:, :, None, None]
        if bv is not None:
            z = z.masked_fill(~bv, 0.0)
        return self.norm(self.stem(z, wavelengths_nm, bv)[:, :, 0, 0])

    def forward_pixel(self, x, r, wavelengths_nm, band_valid=None):
        """(embedding (P, E) unit, level-1 logit (P,), fill (P,))."""
        zx, zr = self.encode(x, wavelengths_nm, band_valid), self.encode(r, wavelengths_nm, band_valid)
        h = self.pixel(torch.cat([zx, zr, zx - zr], dim=1))
        return F.normalize(self.to_embed(h), dim=1), self.to_manmade(h)[:, 0], torch.sigmoid(self.to_fill(h)[:, 0])

    def forward_library(self, spectra, wavelengths_nm):
        """(L, N) library spectra -> (L, E) unit embeddings."""
        return F.normalize(self.library(self.encode(spectra, wavelengths_nm)), dim=1)

    def class_scores(self, z, lib_embed, lib_cls, n_classes: int, exclude=None):
        """(P, C) logsumexp over each class's entries of cos / tau; -inf for a class with no entry (lib_cls -1 is
        no class). exclude (P,) entry index per sample to leave out (its own planted spectrum), -1 for none."""
        s = z @ lib_embed.T / self.log_tau.exp()                           # (P, L)
        if exclude is not None:
            rows = torch.nonzero(exclude >= 0).squeeze(1)
            s = s.index_put((rows, exclude[rows]), torch.tensor(float("-inf"), device=s.device, dtype=s.dtype))
        out = torch.full((len(z), n_classes), float("-inf"), device=z.device, dtype=s.dtype)
        for c in range(n_classes):
            cols = lib_cls == c
            if cols.any():
                out[:, c] = torch.logsumexp(s[:, cols], dim=1)
        return out

    @classmethod
    def from_spectral_stem(cls, state_dict: dict, **kw) -> "SpectralNamer":
        """A namer whose stats, stem and post-stem norm are a spectral-stem checkpoint's (Indradhanu Heavy)."""
        m = cls(state_dict["stats.wl"].tolist(), state_dict["stats.mean"].tolist(), state_dict["stats.std"].tolist(),
                **kw)
        for prefix, mod in (("spectral_in.", m.stem), ("norm.", m.norm)):
            mod.load_state_dict({k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)})
        return m
