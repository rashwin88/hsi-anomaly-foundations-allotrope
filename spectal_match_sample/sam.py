"""
Spectral Angle Mapper (SAM).

For each unknown spectrum u and library spectrum l, the spectral angle is

    theta = arccos( (u . l) / (||u|| ||l||) )

with theta in [0, pi/2] for non-negative reflectance vectors. Smaller is
more similar. We return theta in DEGREES (more interpretable than radians)
because the literature uses degrees and the user's confidence cuts will
be expressed that way.

Why SAM and not NS3
-------------------
SAM is amplitude-invariant: scaling u by any positive factor leaves theta
unchanged. That's exactly what we want when comparing lab/field reflectance
(typically higher amplitude, illumination-controlled) against atmospherically-
corrected spaceborne reflectance (lower amplitude, residual atmospheric
effects). NS3 deliberately couples shape and amplitude, which is the wrong
weighting for a cross-platform match.

Vectorisation
-------------
For P unknowns x N library entries, SAM is

    cos_theta = (U_norm @ L_norm.T)   shape (P, N)

where U_norm = U / ||U|| (per row), L_norm = L / ||L||. This is a single
matmul. A 5000 x 149 by 149 x 2400 matmul:
    - on a Colab CPU it takes ~1-3 seconds
    - on a Colab T4 GPU it takes ~30 ms

The dispatch picks GPU automatically when CUDA is available and falls back
to CPU when it isn't, so the same code runs in both environments.

Memory note
-----------
The output (P, N) float32 matrix at 5000 x 2500 is 50 MB. At 100k pixels
it's 1 GB, which is fine on CPU but tight on smaller GPUs. We chunk along
the pixel axis when the projected output exceeds `chunk_threshold_bytes`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _resolve_device(device: str) -> str:
    """Translate 'auto' to 'cuda' or 'cpu' based on torch availability."""
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device='cuda' requested but torch.cuda.is_available() is False. "
                    "Either install CUDA-enabled torch or use device='auto'/'cpu'."
                )
        except ImportError as exc:
            raise RuntimeError(
                "device='cuda' requested but PyTorch is not installed."
            ) from exc
        return "cuda"
    if device == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {device!r}; expected 'auto'|'cpu'|'cuda'")


def _l2_normalize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-row L2 normalisation. NumPy version."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _sam_numpy(unknown: np.ndarray, library: np.ndarray) -> np.ndarray:
    u = _l2_normalize(unknown.astype(np.float32, copy=False))
    l = _l2_normalize(library.astype(np.float32, copy=False))
    cos_theta = np.clip(u @ l.T, -1.0, 1.0)
    # SAM in radians, then convert to degrees for downstream interpretability.
    theta_rad = np.arccos(cos_theta)
    return np.rad2deg(theta_rad).astype(np.float32, copy=False)


def _sam_torch(
    unknown: np.ndarray,
    library: np.ndarray,
    device: str,
    chunk_pixels: int,
) -> np.ndarray:
    import torch

    P = unknown.shape[0]
    N = library.shape[0]

    L_t = torch.from_numpy(library.astype(np.float32, copy=False)).to(device)
    L_t = torch.nn.functional.normalize(L_t, p=2, dim=1)        # (N, B)
    L_T = L_t.t().contiguous()                                  # (B, N)

    out = np.empty((P, N), dtype=np.float32)
    for s in range(0, P, chunk_pixels):
        e = min(s + chunk_pixels, P)
        U_t = torch.from_numpy(
            unknown[s:e].astype(np.float32, copy=False)
        ).to(device)
        U_t = torch.nn.functional.normalize(U_t, p=2, dim=1)    # (chunk, B)
        cos_t = torch.clamp(U_t @ L_T, -1.0, 1.0)               # (chunk, N)
        theta = torch.rad2deg(torch.arccos(cos_t))
        out[s:e] = theta.detach().cpu().numpy()
    return out


def compute_sam_matrix(
    unknown: np.ndarray,
    library: np.ndarray,
    device: str = "auto",
    chunk_threshold_bytes: int = 1 << 30,   # 1 GB
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the SAM angle (in DEGREES) between every unknown and every library
    entry.

    Parameters
    ----------
    unknown : (P, B) ndarray
        P unknown spectra, B bands.
    library : (N, B) ndarray
        N library spectra, same B bands.
    device : 'auto' | 'cpu' | 'cuda'
        Compute device.
    chunk_threshold_bytes : int
        If the full output matrix would exceed this, chunk along the pixel
        axis to bound memory.
    verbose : bool
        Print device and timing.

    Returns
    -------
    np.ndarray
        (P, N) float32 angles in degrees, in [0, 90] for non-negative inputs.
    """
    if unknown.ndim != 2 or library.ndim != 2:
        raise ValueError(
            f"expected 2-D inputs, got {unknown.shape} and {library.shape}"
        )
    if unknown.shape[1] != library.shape[1]:
        raise ValueError(
            f"band count mismatch: unknown has {unknown.shape[1]}, "
            f"library has {library.shape[1]}. "
            "Library must be resampled to the same target wavelengths."
        )

    P, B = unknown.shape
    N = library.shape[0]
    out_bytes = P * N * 4

    resolved = _resolve_device(device)
    if verbose:
        print(
            f"[sam] computing ({P} x {N}) on device={resolved}, "
            f"output={out_bytes / 1e6:.1f} MB"
        )

    if resolved == "cpu":
        return _sam_numpy(unknown, library)

    # GPU path: chunk if the projected output is large.
    if out_bytes > chunk_threshold_bytes:
        chunk_rows = max(1, chunk_threshold_bytes // (N * 4))
    else:
        chunk_rows = P
    return _sam_torch(unknown, library, device=resolved, chunk_pixels=chunk_rows)
