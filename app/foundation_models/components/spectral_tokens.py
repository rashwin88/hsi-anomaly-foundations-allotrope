"""
Turn a band's wavelength into a vector, so a spectrum can be read as tokens that know where
they sit in the spectrum.

Used by the cross-attention compressor and decoder (spectral_attention.py), which tag every
band token - and every requested output band - with this encoding.

Why wavelength and not band index: an index ties the model to one band list ("input 87"),
so a model trained on SpectralEarth's 202 bands could not read the 165-band grid or PRISMA's
native bands. A wavelength means the same thing on every sensor, and nearby wavelengths get
similar vectors, so the model can carry what it learned at 1650 nm to a band at 1655 nm.
Design: docs/proposals/spectral-attention-bottleneck.md, decision 2; taught in
`indradhanu workings/33-a-spectrum-as-tokens.md`.

Gotchas: wavelengths in nanometres; output (..., dim) with the sines first, then the cosines.
"""

import math

import torch

# Shortest and longest period, nm. 10 nm resolves neighbouring bands (EnMAP and PRISMA sample
# every ~6-12 nm); 10,000 nm exceeds the whole 400-2500 nm range, so the slowest pair varies
# monotonically across it and acts as a coarse "where in the spectrum" coordinate.
MIN_PERIOD_NM, MAX_PERIOD_NM = 10.0, 10_000.0


def wavelength_encoding(wavelengths_nm: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal encoding: (...) wavelengths in nm -> (..., dim), dim/2 periods spaced
    geometrically from 10 to 10,000 nm. Raises ValueError unless dim is even and >= 4."""
    if dim % 2 or dim < 4:
        raise ValueError(f"dim must be even and >= 4, got {dim}")
    half = dim // 2
    i = torch.arange(half, dtype=torch.float32, device=wavelengths_nm.device)
    period = MIN_PERIOD_NM * (MAX_PERIOD_NM / MIN_PERIOD_NM) ** (i / (half - 1))
    angle = 2 * math.pi * wavelengths_nm.float()[..., None] / period
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
