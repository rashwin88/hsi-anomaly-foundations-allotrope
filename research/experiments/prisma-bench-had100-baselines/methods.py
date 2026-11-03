"""
HAD100's classical comparison methods (Li et al., IEEE TGRS 2023, Table V) re-implemented for full PRISMA scenes on
the GPU. Research only; nothing here is product code.

    grx(X, valid)                     global RX, ridge-regularised covariance
    lrx(X, valid, inner, outer)       local (dual-window) RX: ring mean and ring covariance per pixel, ridge
    crd(Z, valid, inner, outer, lam)  collaborative representation detector (Li & Du 2015), closed form
    kcsr(Z, valid, inner, outer)      kernel constrained sparse representation (Ling et al. 2019), approximated

X is (C, H, W) float32 reflectance; Z a (k, H, W) projection (see second_moment_projection). Windows are full odd
widths; the ring is outer minus inner box. Invalid pixels never enter a ring. Scores are 0 outside valid.

Why the projection for CR / KCSR: both solve one regression per pixel over its ring (816 samples at 5/29). On 185
bands that is ~2e13 flops per scene for CR and an 832 x 832 kernel per pixel for KCSR - hours per scene. Both use only
Euclidean geometry (inner products, distances), which the top-k eigenvectors of the uncentred second-moment matrix
preserve almost exactly (energy captured is logged). LRX and GRX run on the full bands.
"""
from __future__ import annotations

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def grx(X: np.ndarray, valid: np.ndarray, ridge: float = 1e-6, bands: np.ndarray | None = None) -> np.ndarray:
    """(x - mu)^T (S + ridge * tr(S)/C * I)^-1 (x - mu) with the scene's own mean / covariance (float64)."""
    Xb = X if bands is None else X[bands]
    V = torch.from_numpy(Xb[:, valid].T.copy()).to(DEV, torch.float64)          # (N, C)
    mu = V.mean(0)
    V -= mu
    S = V.T @ V / (len(V) - 1)
    C = S.shape[0]
    S += ridge * torch.trace(S) / C * torch.eye(C, device=DEV, dtype=torch.float64)
    L = torch.linalg.cholesky(S)
    out = np.zeros(valid.shape, np.float32)
    s = torch.empty(len(V), device=DEV, dtype=torch.float64)
    for i in range(0, len(V), 200_000):
        Y = torch.linalg.solve_triangular(L, V[i:i + 200_000].T, upper=False)
        s[i:i + 200_000] = (Y ** 2).sum(0)
    out[valid] = s.float().cpu().numpy()
    return out


def _box_sum(F: torch.Tensor, half: int, pad: int, h: int, w: int) -> torch.Tensor:
    """Sum over the (2 half + 1)^2 box at each of h x w centres from an integral image led by one zero row / col,
    of an input padded by `pad`."""
    a, b = pad - half, pad + half + 1
    return F[:, b:b + h, b:b + w] - F[:, a:a + h, b:b + w] - F[:, b:b + h, a:a + w] + F[:, a:a + h, a:a + w]


def lrx(X: np.ndarray, valid: np.ndarray, inner: int, outer: int, ridge: float = 1e-4, strip: int = 4) -> np.ndarray:
    """Dual-window RX: per pixel, the ring's mean and covariance (outer box minus inner box of valid pixels), ridge
    ridge * tr(S_ring)/C. Moments by integral images of x and x x^T over row strips (data centred on the scene mean,
    float64: in float32 the moment cancellation left 5/15 rings indefinite). Pixels with fewer ring samples than bands + 1 fall back to the global covariance."""
    C, H, W = X.shape
    r = outer // 2
    mu_g = X[:, valid].mean(1)
    Xt = torch.from_numpy(((X - mu_g[:, None, None]) * valid).astype(np.float64)).to(DEV)
    vt = torch.from_numpy(valid.astype(np.float64)).to(DEV)
    iu = torch.triu_indices(C, C, device=DEV)
    Sg = torch.from_numpy(np.cov(X[:, valid][:, ::7])).to(DEV, torch.float64)
    out = torch.zeros(H, W, device=DEV, dtype=torch.float64)
    eye = torch.eye(C, device=DEV, dtype=torch.float64)
    n_bad = [0]
    for r0 in range(0, H, strip):
        r1 = min(H, r0 + strip)
        lo, hi = r0 - r, r1 + r                       # padded input rows
        h = r1 - r0
        def rows(T):
            pad_top, pad_bot = max(0, -lo), max(0, hi - H)
            T = T[..., max(lo, 0):min(hi, H), :]
            return torch.nn.functional.pad(T, (r, r, pad_top, pad_bot))
        x = rows(Xt)                                   # (C, h+2r, W+2r)
        m = rows(vt[None])                             # (1, ...)
        Kc = 1 + C + len(iu[0])
        F = torch.zeros(Kc, x.shape[1] + 1, x.shape[2] + 1, device=DEV, dtype=torch.float64)   # integral image, zero-led
        F[0, 1:, 1:] = m[0]
        F[1:C + 1, 1:, 1:] = x
        o = 1 + C
        for i in range(C):                             # upper-triangle products, row i: x_i * x_j, j >= i
            F[o:o + C - i, 1:, 1:] = x[i] * x[i:]
            o += C - i
        F.cumsum_(1)
        F.cumsum_(2)
        A = _box_sum(F, outer // 2, r, h, W) - _box_sum(F, inner // 2, r, h, W)
        del F
        n = A[0].reshape(-1)
        s1 = A[1:C + 1].reshape(C, -1).T                # (h W, C)
        s2 = A[C + 1:].reshape(len(iu[0]), -1).T
        del A
        M = torch.zeros(h * W, C, C, device=DEV, dtype=torch.float64)
        M[:, iu[0], iu[1]] = s2
        M[:, iu[1], iu[0]] = s2
        del s2
        nn = n.clamp(min=2)[:, None]
        mu = s1 / nn
        S = (M - nn[:, :, None] * mu[:, :, None] * mu[:, None, :]) / (nn[:, :, None] - 1)
        del M
        few = n < C + 1
        S[few] = Sg
        tr = torch.diagonal(S, dim1=1, dim2=2).sum(1) / C
        S += (ridge * tr)[:, None, None] * eye
        y = Xt[:, r0:r1, :].reshape(C, -1).T - mu
        y[few] = Xt[:, r0:r1, :].reshape(C, -1).T[few]
        L, info = torch.linalg.cholesky_ex(S)
        z = torch.linalg.solve_triangular(L, y[:, :, None], upper=False)[:, :, 0]
        q = (z ** 2).sum(1)
        bad = (info > 0) | ~torch.isfinite(q)
        if bad.any():          # still indefinite: redo with 10x the ridge
            Sb = S[bad] + (9 * ridge * tr[bad])[:, None, None] * eye
            q[bad] = (y[bad][:, None, :] @ torch.linalg.solve(Sb, y[bad][:, :, None]))[:, 0, 0]
            n_bad[0] += int(bad.sum())
        out[r0:r1] = q.reshape(h, W)
        del S, L
    lrx.last_refit = n_bad[0]
    return (out.cpu().numpy() * valid).astype(np.float32)


def second_moment_projection(X: np.ndarray, valid: np.ndarray, k: int = 30) -> tuple[np.ndarray, float]:
    """Project uncentred spectra onto the top-k eigenvectors of E[x x^T] over valid pixels: (k, H, W), and the
    share of total energy (sum of x^2) kept."""
    V = torch.from_numpy(X[:, valid][:, ::5].T.copy()).to(DEV, torch.float64)
    M = V.T @ V / len(V)
    ev, U = torch.linalg.eigh(M)
    U = U[:, -k:].flip(1)
    kept = float(ev[-k:].sum() / ev.sum())
    Ut = U.float()
    Z = torch.einsum("ck,chw->khw", Ut, torch.from_numpy(X).to(DEV)).cpu().numpy() * valid
    return Z.astype(np.float32), kept


def ring_offsets(inner: int, outer: int, stride: int = 1) -> np.ndarray:
    r, q = outer // 2, inner // 2
    d = np.arange(-r, r + 1, stride)
    oy, ox = np.meshgrid(d, d, indexing="ij")
    keep = (np.abs(oy) > q) | (np.abs(ox) > q)
    return np.stack([oy[keep], ox[keep]], 1)


def _gather_ring(Zp: torch.Tensor, vp: torch.Tensor, ys: torch.Tensor, xs: torch.Tensor, off: torch.Tensor, r: int):
    """Ring samples (B, n, k) and their validity (B, n) for centres (ys, xs) in an array padded by r."""
    yy = ys[:, None] + r + off[None, :, 0]
    xx = xs[:, None] + r + off[None, :, 1]
    return Zp[:, yy, xx].permute(1, 2, 0), vp[yy, xx]


def crd(Z: np.ndarray, valid: np.ndarray, inner: int = 5, outer: int = 29, lams=(1e-2,), batch: int = 6144) -> dict:
    """CRD (Li & Du 2015): alpha = argmin ||y - X a||^2 + lam ||Gamma_y a||^2, Gamma_y = diag(||y - x_i||); score
    ||y - X alpha||. With D = lam Gamma^2 and G = X D^-1 X^T (k x k), the residual is exactly (G + I)^-1 y
    (push-through identity), so each pixel costs a k x k solve. Returns {lam: (H, W) score}."""
    k, H, W = Z.shape
    r = outer // 2
    off = torch.from_numpy(ring_offsets(inner, outer)).to(DEV)
    Zp = torch.nn.functional.pad(torch.from_numpy(Z).to(DEV), (r, r, r, r))
    vp = torch.nn.functional.pad(torch.from_numpy(valid).to(DEV), (r, r, r, r))
    pix = torch.from_numpy(np.argwhere(valid)).to(DEV)
    res = {lam: torch.zeros(H, W, device=DEV) for lam in lams}
    eye = torch.eye(k, device=DEV)
    for i in range(0, len(pix), batch):
        ys, xs = pix[i:i + batch, 0], pix[i:i + batch, 1]
        R, m = _gather_ring(Zp, vp, ys, xs, off, r)                     # (B, n, k), (B, n)
        y = Zp[:, ys + r, xs + r].T                                      # (B, k)
        d2 = ((R - y[:, None]) ** 2).sum(-1).clamp(min=1e-10)            # (B, n)
        for lam in lams:
            w = m / (lam * d2)
            G = torch.einsum("bn,bni,bnj->bij", w, R, R)
            e = torch.linalg.solve(G + eye, y[:, :, None])[:, :, 0]
            res[lam][ys, xs] = e.norm(dim=1)
    return {lam: v.cpu().numpy().astype(np.float32) for lam, v in res.items()}


def _proj_simplex(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each row onto {a >= 0, sum a = 1, a = 0 where mask is False}."""
    v = torch.where(mask, v, torch.full_like(v, -1e9))
    u, _ = torch.sort(v, dim=1, descending=True)
    css = u.cumsum(1) - 1
    j = torch.arange(1, v.shape[1] + 1, device=v.device)
    cond = (u - css / j) > 0
    rho = cond.float().cumsum(1).argmax(1, keepdim=True)            # last index where cond holds
    theta = css.gather(1, rho) / (rho + 1)
    return (v - theta).clamp(min=0)


def kcsr(Z: np.ndarray, valid: np.ndarray, inner: int = 3, outer: int = 29, stride: int = 2, iters: int = 100,
         batch: int = 4096) -> np.ndarray:
    """Kernel constrained sparse representation (Ling et al. 2019), approximated: min over the simplex
    (a >= 0, sum a = 1) of ||phi(y) - Phi a||^2 with an RBF kernel, sigma^2 = mean squared distance from y to its
    ring; solved by projected gradient (step 1 / max row sum of K). Score = the feature-space residual
    k(y,y) - 2 k_y^T a + a^T K a. Approximations: ring subsampled on a stride-2 lattice (224 of 832 samples at
    3/29), top-k projection, fixed iterations."""
    k, H, W = Z.shape
    r = outer // 2
    off = torch.from_numpy(ring_offsets(inner, outer, stride)).to(DEV)
    Zp = torch.nn.functional.pad(torch.from_numpy(Z).to(DEV), (r, r, r, r))
    vp = torch.nn.functional.pad(torch.from_numpy(valid).to(DEV), (r, r, r, r))
    pix = torch.from_numpy(np.argwhere(valid)).to(DEV)
    out = torch.zeros(H, W, device=DEV)
    for i in range(0, len(pix), batch):
        ys, xs = pix[i:i + batch, 0], pix[i:i + batch, 1]
        R, m = _gather_ring(Zp, vp, ys, xs, off, r)
        y = Zp[:, ys + r, xs + r].T
        mf = m.float()
        dy = ((R - y[:, None]) ** 2).sum(-1)
        s2 = ((dy * mf).sum(1) / mf.sum(1).clamp(min=1)).clamp(min=1e-10)[:, None]
        ky = torch.exp(-dy / (2 * s2)) * mf
        sq = (R ** 2).sum(-1)
        DD = (sq[:, :, None] + sq[:, None, :] - 2 * R @ R.transpose(1, 2)).clamp(min=0)
        K = torch.exp(-DD / (2 * s2[:, :, None])) * mf[:, :, None] * mf[:, None, :]
        L = K.sum(2).amax(1, keepdim=True).clamp(min=1e-6)
        a = mf / mf.sum(1, keepdim=True).clamp(min=1)
        for _ in range(iters):
            g = (K @ a[:, :, None])[:, :, 0] - ky
            a = _proj_simplex(a - g / L, m)
        e = 1 - 2 * (ky * a).sum(1) + (a[:, None, :] @ K @ a[:, :, None])[:, 0, 0]
        out[ys, xs] = e.clamp(min=0)
    return out.cpu().numpy().astype(np.float32)
