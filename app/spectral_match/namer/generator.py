"""
Planted samples for the namer, built on the GPU from real tiles (docs/proposals/spectral-namer.md).

For a batch of tiles (B, N, H, W) with band validity and one band list:
    ring mean r: mean of valid pixels 3 to 7 px away in Chebyshev distance, 200 of them (box sums of 15 x 15 minus 5 x 5)
    noise sigma: per tile and band, RMS (uncentred) of horizontal neighbour differences / sqrt(2) - texture counts too
Then per sampled pixel b (and its r):
    kind 0 man-made     t a man-made library spectrum (train split), fill a log-uniform in [a_min, 1]
    kind 1 natural-mix  t a natural library spectrum, same fill
    kind 2 plain        a = 0
    x  = g (a t + (1 - a) b) + n,   r' = g r,   g = 1 + gain * smooth curve in [-1, 1],   n ~ N(0, sigma)
Returns x, r', band validity, man-made flag, class (-1 if not man-made), entry index, fill.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

INNER, OUTER = 5, 15                                     # full box widths: ring 3..7 px


def ring_and_noise(pixels: torch.Tensor, band_valid: torch.Tensor):
    """pixels (B, N, H, W), band_valid (B, N, H, W) -> ring mean (B, N, H, W), ring count (B, 1, H, W),
    noise sigma (B, N)."""
    m = band_valid.float()
    v = pixels * m
    def box(t, k):
        return F.avg_pool2d(t, k, stride=1, padding=k // 2, count_include_pad=True) * (k * k)
    s = box(v, OUTER) - box(v, INNER)
    c = box(m, OUTER) - box(m, INNER)
    ring = s / c.clamp(min=1)
    count = c.min(dim=1, keepdim=True).values
    d = (pixels[..., 1:] - pixels[..., :-1]) * (m[..., 1:] * m[..., :-1])
    n = (m[..., 1:] * m[..., :-1]).sum(dim=(2, 3)).clamp(min=2)
    sigma = ((d ** 2).sum(dim=(2, 3)) / n).sqrt() / 2 ** 0.5
    return ring, count, sigma


def smooth_gain(n_samples: int, n_bands: int, gain: float, device, generator=None) -> torch.Tensor:
    """(P, N) 1 + gain * a smooth (quadratic, clipped) curve over the band index."""
    t = torch.linspace(-1, 1, n_bands, device=device)
    c = torch.randn(n_samples, 3, device=device, generator=generator) * 0.5
    curve = (c[:, :1] + c[:, 1:2] * t + c[:, 2:3] * t * t).clamp(-1, 1)
    return 1 + gain * curve


def warp(spectra: torch.Tensor, wl: torch.Tensor, shift_nm: torch.Tensor, stretch: torch.Tensor) -> torch.Tensor:
    """(P, N) spectra read at wl * (1 + stretch) + shift_nm (per row), by linear interpolation along wavelength
    (ends held); wl in any order."""
    order = torch.argsort(wl)
    w, s = wl[order], spectra[:, order]
    q = (w[None] * (1 + stretch[:, None]) + shift_nm[:, None]).clamp(w[0], w[-1])
    hi = torch.searchsorted(w, q.contiguous()).clamp(1, len(w) - 1)
    lo = hi - 1
    f = ((q - w[lo]) / (w[hi] - w[lo]).clamp(min=1e-6)).clamp(0, 1)
    out = torch.lerp(s.gather(1, lo), s.gather(1, hi), f)
    return out[:, torch.argsort(order)]


def augment_spectra(t: torch.Tensor, wl: torch.Tensor, aug: dict, generator=None) -> torch.Tensor:
    """Library-spectrum augmentation (N003): a small wavelength shift / stretch and a smooth multiplicative tilt -
    plausible variants of a material, so the model cannot memorise ~125 fixed spectra."""
    P, N = t.shape
    dev = t.device
    u = lambda a: (torch.rand(P, device=dev, generator=generator) * 2 - 1) * a
    out = warp(t, wl, u(aug.get("shift_nm", 0.0)), u(aug.get("stretch", 0.0)))
    return out * smooth_gain(P, N, aug.get("tilt", 0.0), dev, generator)


def mix_within_class(entry: torch.Tensor, labels: torch.Tensor, refl: torch.Tensor, pools: dict, p: float,
                     generator=None) -> torch.Tensor:
    """(P, N) spectra: refl[entry], and with probability p a convex mix w t_i + (1 - w) t_j with a partner j of the
    same label from the same pool (w ~ U(0.5, 1)). labels: class per entry (natural = its own label)."""
    dev = refl.device
    t = refl[entry.clamp(min=0)]
    P = len(entry)
    if p <= 0:
        return t
    partner = entry.clone()
    for lab, pool in pools.items():
        rows = torch.nonzero((entry >= 0) & (labels[entry.clamp(min=0)] == lab)).squeeze(1)
        if len(rows) and len(pool) > 1:
            partner[rows] = pool[torch.randint(len(pool), (len(rows),), device=dev, generator=generator)]
    w = 0.5 + 0.5 * torch.rand(P, 1, device=dev, generator=generator)
    mixed = w * t + (1 - w) * refl[partner.clamp(min=0)]
    use = (torch.rand(P, device=dev, generator=generator) < p) & (entry >= 0)
    return torch.where(use[:, None], mixed, t)


def plant(pixels, band_valid, bank, per_tile: int = 256, a_min: float = 0.05, gain: float = 0.1,
          kinds=(0.5, 0.25, 0.25), split: str = "train", fixed_fill: float | None = None, generator=None,
          wl: torch.Tensor | None = None, lib_aug: dict | None = None):
    """Planted samples from one band list's tiles. bank: namer.bank.LibraryBank on the same device and band list."""
    device = pixels.device
    B, N, H, W = pixels.shape
    ring, count, sigma = ring_and_noise(pixels, band_valid)
    ok = band_valid.all(dim=1) & (count[:, 0] >= 20)                       # every band valid, enough ring pixels
    ok[:, :OUTER // 2] = ok[:, -(OUTER // 2):] = False
    ok[:, :, :OUTER // 2] = ok[:, :, -(OUTER // 2):] = False
    idx = []
    for b in range(B):
        flat = torch.nonzero(ok[b].flatten()).squeeze(1)
        if len(flat):
            pick = flat[torch.randint(len(flat), (per_tile,), device=device, generator=generator)]
            idx.append(torch.stack([torch.full_like(pick, b), pick // W, pick % W], 1))
    if not idx:
        return None
    idx = torch.cat(idx)
    bi, yi, xi = idx[:, 0], idx[:, 1], idx[:, 2]
    P = len(idx)
    bkg = pixels[bi, :, yi, xi]                                            # (P, N)
    rmean = ring[bi, :, yi, xi]
    sig = sigma[bi]
    kind = torch.multinomial(torch.tensor(kinds, device=device), P, replacement=True, generator=generator)
    unseen = bank.test | (bank.cls == bank.holdout_class) if bank.holdout_class >= 0 else bank.test
    mm_pool = torch.nonzero(bank.manmade & (unseen if split == "test" else ~unseen)).squeeze(1)
    nat_pool = torch.nonzero(~bank.manmade).squeeze(1)                    # natural spectra: both splits
    entry = torch.full((P,), -1, dtype=torch.long, device=device)
    is_mm, is_nat = kind == 0, kind == 1
    if len(mm_pool):
        entry[is_mm] = mm_pool[torch.randint(len(mm_pool), (int(is_mm.sum()),), device=device, generator=generator)]
    else:
        is_mm = torch.zeros_like(is_mm)
    entry[is_nat] = nat_pool[torch.randint(len(nat_pool), (int(is_nat.sum()),), device=device, generator=generator)]
    u = torch.rand(P, device=device, generator=generator)
    fill = torch.exp(torch.log(torch.tensor(a_min)) * (1 - u)).to(device)  # log-uniform in [a_min, 1]
    if fixed_fill is not None:                                             # evaluation at one fill
        fill = torch.full_like(fill, fixed_fill)
    fill = torch.where(is_mm | is_nat, fill, torch.zeros_like(fill))
    if lib_aug and split == "train" and wl is not None:                   # N003: varied library spectra
        labels = torch.where(bank.manmade, bank.cls, torch.full_like(bank.cls, -1))
        pools = {int(c): mm_pool[bank.cls[mm_pool] == c] for c in bank.cls[mm_pool].unique()}
        pools[-1] = nat_pool
        t = augment_spectra(mix_within_class(entry, labels, bank.refl, pools, lib_aug.get("mix_p", 0.0), generator),
                            wl, lib_aug, generator)
        t = torch.where((entry >= 0)[:, None], t, torch.zeros_like(bkg))
    else:
        t = torch.where((entry >= 0)[:, None], bank.refl[entry.clamp(min=0)], torch.zeros_like(bkg))
    g = smooth_gain(P, N, gain, device, generator)
    x = g * (fill[:, None] * t + (1 - fill[:, None]) * bkg) + torch.randn(P, N, device=device, generator=generator) * sig
    r = g * rmean
    cls = torch.where(is_mm, bank.cls[entry.clamp(min=0)], torch.full_like(entry, -1))
    return dict(x=x, r=r, manmade=is_mm.float(), cls=cls, entry=entry, fill=fill)
