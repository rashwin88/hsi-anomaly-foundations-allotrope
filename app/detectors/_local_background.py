"""
Shared machinery for the local-background (annulus) detectors.

LocalRXDetector and MNFCompressionLRXDetector both score each pixel against a
ring of its own neighbours rather than against the whole scene, and both need
the same two things to do it: a device, and a way to compute thousands of small
Mahalanobis distances without a Python loop. These functions lived duplicated,
byte-for-byte, in each detector.

Why the batched form is necessary rather than nice-to-have: local RX needs one
covariance matrix, one inverse and one quadratic form PER PIXEL. A megapixel
scene means a million small linear solves, which is hopeless one at a time in
numpy. Batching them into a single (N, B, B) tensor turns it into one call that
a GPU can saturate.
"""

from __future__ import annotations

import numpy as np
import torch


def select_device() -> torch.device:
    """Pick the best available torch device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batch_mahalanobis(
    X_bg_padded: np.ndarray,   # (N, max_bg, B) float64
    n_bg_arr: np.ndarray,      # (N,) int64 - actual bg count per pixel
    x_test: np.ndarray,        # (N, B) float64 - test pixel spectra
    count: int,                # how many entries in this batch are valid
    B: int,
    reg: float,
    device: torch.device,
) -> np.ndarray:
    """
    Batched covariance + solve + Mahalanobis on device. Returns (count,) scores.

    Background counts vary per pixel - a pixel near an edge or beside an invalid
    region has fewer valid neighbours - so `X_bg_padded` is padded to the widest
    case and `n_bg_arr` says how much of each row is real. Everything below is
    masked accordingly; padding contributes nothing to the mean or covariance.
    """
    # float32 on GPU for speed; float64 on CPU for precision.
    dtype = torch.float32 if device.type != "cpu" else torch.float64

    X = torch.from_numpy(X_bg_padded[:count]).to(device=device, dtype=dtype)
    n = torch.from_numpy(n_bg_arr[:count]).to(device=device)
    xt = torch.from_numpy(x_test[:count]).to(device=device, dtype=dtype)

    max_bg = X.shape[1]
    # Validity mask: (count, max_bg) - True for real background entries
    indices = torch.arange(max_bg, device=device).unsqueeze(0)
    mask = indices < n.unsqueeze(1)

    # Masked mean: zero out padding, sum, divide by real count
    X_masked = X * mask.unsqueeze(-1)
    counts_f = mask.sum(dim=1, keepdim=True).to(dtype)       # (count, 1)
    mu = X_masked.sum(dim=1) / counts_f                       # (count, B)

    # Centred data (padding stays zero)
    dX = (X - mu.unsqueeze(1)) * mask.unsqueeze(-1)           # (count, max_bg, B)

    # Batched covariance: (count, B, B)
    cov = dX.transpose(-1, -2) @ dX
    cov = cov / (counts_f.unsqueeze(-1) - 1)
    # Ridge term. A local background is small and often rank-deficient, so the
    # raw covariance is frequently singular; without this the solve blows up.
    cov += reg * torch.eye(B, device=device, dtype=dtype)

    # Test vector relative to its local mean
    x = xt - mu                                                # (count, B)

    # Batched solve: cov @ sol = x  ->  sol = cov^-1 x. Solving beats forming
    # the inverse explicitly - better conditioned and faster.
    sol = torch.linalg.solve(cov, x)                           # (count, B)
    scores = (x * sol).sum(dim=1)                              # (count,)

    return scores.cpu().to(torch.float64).numpy()
