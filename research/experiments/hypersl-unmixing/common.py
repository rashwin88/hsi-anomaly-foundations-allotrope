"""
Shared plumbing for the HyperSL unmixing study (README.md): dataset loaders, the scene normalisation frame, a frozen
HyperSL encoder, and a batched fully-constrained least-squares solver.

Run in WSL (venv ~/hypersl_eval/.venv). Data: local copies of the public unmixing sets under DATA (read-only source:
G:/My Drive/PS_11/claude_experiments/eval_datasets). HyperSL: upstream clone + HF 10_base_mask95 checkpoint.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
import torch

DATA = os.environ.get("HSU_DATA", "/mnt/c/es/hsu/data")
UPSTREAM = "/mnt/c/hypersl_eval/upstream"
CKPT = "/mnt/c/hypersl_eval/10_base_mask95_checkpoint.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Scene:
    name: str
    Y: np.ndarray        # (N, B) reflectance, pixels in row-major (H, W) order
    H: int
    W: int
    wl: np.ndarray       # (B,) nm
    E: np.ndarray        # (K, B) ground-truth endmembers
    A: np.ndarray        # (K, N) ground-truth abundances
    em_names: list
    note: str = ""

    @property
    def cube(self):      # (H, W, B)
        return self.Y.reshape(self.H, self.W, -1)


def _drop(n, ranges):
    keep = np.ones(n, bool)
    for lo, hi in ranges:
        keep[lo - 1:hi] = False
    return keep


def _pick_order(Yb, A, E, H, W):
    """Yb (B, N) in an unknown pixel order and scale -> (N, B) in A's pixel order and E's units: the order (and least-
    squares scalar) for which E @ A reconstructs Y best."""
    best = None
    for order in ("C", "F"):
        Y = Yb.T.reshape(H, W, -1) if order == "C" else Yb.T.reshape(W, H, -1).transpose(1, 0, 2)
        Y = Y.reshape(H * W, -1)
        R = A.T @ E
        s = float((Y * R).sum() / (R * R).sum())
        err = np.abs(Y / s - R).mean()
        if best is None or err < best[0]:
            best = (err, Y / s, f"{order}, /{s:.4g}")
    return best[1], best[0], best[2]


def load(name: str) -> Scene:
    if name.startswith("DC"):
        d = sio.loadmat(f"{DATA}/{name}.mat")
        H, W = int(np.squeeze(d["H"])), int(np.squeeze(d["W"]))
        E = d["E"].astype(np.float64).T                                    # (K, B)
        A = d["A"].astype(np.float64)                                      # (K, N)
        Y, err, o = _pick_order(d["Y"].astype(np.float64), A, E, H, W)
        wl = np.linspace(400, 2500, Y.shape[1])
        names = [f"mineral_{i + 1}" for i in range(len(E))]
        return Scene(name, Y, H, W, wl, E, A, names, f"HySUPP synthetic; |Y-EA| {err:.4f} ({o})")
    if name == "EGU":
        d = sio.loadmat(f"{DATA}/synthetic_data_1.mat")
        H, W = int(np.squeeze(d["m"])), int(np.squeeze(d["n"]))
        E = d["M"].astype(np.float64).T
        A = d["A_true"].astype(np.float64).reshape(H * W, -1).T
        Y, err, o = _pick_order(d["X"].astype(np.float64).reshape(H * W, -1).T, A, E, H, W)
        wl = np.linspace(400, 2500, Y.shape[1])
        names = [f"em_{i + 1}" for i in range(len(E))]
        return Scene(name, Y, H, W, wl, E, A, names, f"EGU-Net synthetic; |Y-EA| {err:.4f} ({o})")
    sub = "Samson_upc" if name == "Samson" else name
    d = sio.loadmat(f"{DATA}/{sub}.mat")
    H, W = int(np.squeeze(d["lines"])), int(np.squeeze(d["cols"]))
    Yb = d["Y"].astype(np.float64)
    E = d["GT"].astype(np.float64)
    S = d["S_GT"].astype(np.float64)                                       # (H, W, K)
    A = S.reshape(H * W, -1).T
    Y, err, o = _pick_order(Yb, A, E, H, W)
    if name == "Samson":
        wl, names = np.linspace(401, 889, 156), ["soil", "tree", "water"]
    elif name == "Jasper":
        wl = np.linspace(380, 2500, 224)[_drop(224, [(1, 3), (108, 112), (154, 166), (220, 224)])]
        names = ["tree", "water", "soil", "road"]
    else:
        wl = np.linspace(400, 2500, 210)[_drop(210, [(1, 4), (76, 76), (87, 87), (101, 111), (136, 153), (198, 210)])]
        names = ["asphalt", "grass", "tree", "roof"]
    return Scene(name, Y, H, W, wl, E, A, names, f"FCLS-derived GT (grade D); |Y-EA| {err:.4f} ({o})")


DATASETS = ["DC1", "DC2", "DC3", "EGU", "Samson", "Jasper", "Urban4"]
H_DATASETS = ["DC1", "DC2", "DC3", "EGU", "Jasper", "Urban4"]   # Samson: the available GT does not reconstruct Y (README)


class Frame:
    """The scene normalisation frame: per-band min-max over the scene's pixels (sklearn minmax_scale, as HyperSL was
    pretrained). Affine per band, so it maps convex mixtures to the same convex mixtures."""

    def __init__(self, Y):
        self.lo = Y.min(0)
        self.span = np.maximum(Y.max(0) - self.lo, 1e-8)

    def __call__(self, X):
        return ((X - self.lo) / self.span).astype(np.float32)


_MODEL = None


def hypersl():
    global _MODEL
    if _MODEL is None:
        sys.path.insert(0, UPSTREAM)
        from engine.model import SpectralSharedEncoder
        m = SpectralSharedEncoder(embedding_dim=256, encoder_depth=8, decoder_depth=4, num_heads=8)
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
        m.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()}, strict=True)
        _MODEL = m.to(DEV).eval()
    return _MODEL


@torch.no_grad()
def _tokens(m, xb, wl_t):
    """Upstream encoder_forward with mask ratio 0 and without its (permutation-only) shuffle: (n, 1 + B, 256)."""
    x = m.embedding(xb[:, None, :]).transpose(2, 1)
    w = ((wl_t - 400) / 10)[None, :, None]
    pos = torch.zeros(1, x.shape[1], x.shape[2], device=x.device)
    pos[..., 0::2] = torch.sin(m.div_term * w)
    pos[..., 1::2] = torch.cos(m.div_term * w)
    x = torch.cat([m.global_token.expand(len(x), -1, -1), x + pos], 1)
    for blk in m.encoder_blocks:
        x = blk(x)
    return m.transformer_encoder_layer(x)[0]


@torch.no_grad()
def encode(Xn, wl, batch=1024, tokens_fn=None):
    """Xn (N, B) frame-normalised spectra -> pooled z (N, 256) (upstream `aggregate`: global token + mean of band
    tokens). tokens_fn(i0, band_tokens (n, B, 256)) is called per batch for per-token statistics."""
    m = hypersl()
    wl_t = torch.as_tensor(np.asarray(wl), dtype=torch.float32, device=DEV)
    out = []
    for i in range(0, len(Xn), batch):
        xb = torch.as_tensor(Xn[i:i + batch], dtype=torch.float32, device=DEV)
        t = _tokens(m, xb, wl_t)
        out.append((t[:, 0] + t[:, 1:].mean(1)).cpu())
        if tokens_fn is not None:
            tokens_fn(i, t[:, 1:])
    return torch.cat(out).numpy().astype(np.float64)


def fcls(G, b, iters=3000):
    """Batched FCLS: argmin_a 0.5 a'Ga - b'a s.t. a >= 0, sum a = 1 (FISTA with simplex projection).
    G (K, K) Gram of the endmembers, b (N, K) endmember . pixel. Returns (N, K)."""
    G = torch.as_tensor(G, dtype=torch.float64, device=DEV)
    b = torch.as_tensor(b, dtype=torch.float64, device=DEV)
    K = G.shape[0]
    L = torch.linalg.eigvalsh(G).max().clamp(min=1e-12)
    a = torch.full((len(b), K), 1.0 / K, dtype=torch.float64, device=DEV)
    y, t = a.clone(), 1.0
    for _ in range(iters):
        a_new = _simplex(y - (y @ G - b) / L)
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = a_new + (t - 1) / t_new * (a_new - a)
        a, t = a_new, t_new
    return a.cpu().numpy()


def _simplex(v):
    """Row-wise Euclidean projection onto the probability simplex."""
    u, _ = torch.sort(v, dim=1, descending=True)
    css = u.cumsum(1) - 1
    k = torch.arange(1, v.shape[1] + 1, device=v.device, dtype=v.dtype)
    rho = ((u - css / k) > 0).sum(1, keepdim=True)
    theta = css.gather(1, rho - 1) / rho
    return (v - theta).clamp(min=0)


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
