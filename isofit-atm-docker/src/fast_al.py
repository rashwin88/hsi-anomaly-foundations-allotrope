"""Batched per-pixel solve for ISOFIT 4.1.5's analytical line (AC26-BAT, AC26-BSC): same maths, 3.5-7x faster.

apply_oe imports `analytical_line` from here instead of from ISOFIT (build/isofit_patches.py, "fast_al"). It runs
ISOFIT's own analytical_line - atmosphere interpolation, output files, Ray actors, chunk writing - with ISOFIT's
Worker actor swapped for FastWorker, whose run_chunks changes only how the per-pixel posterior is solved:
  - per pixel, in the actor, as ISOFIT does: the algebraic initial guess, then the vectors invert_analytical needs
    (atmosphere terms, theta = diag(H), diagonal Seps, prior component and norm, prior mean);
  - per chunk, batched: A = Sa_inv(component) / norm^2 + diag(theta^2 / Seps) and its right-hand side, solved by
    torch fp64 Cholesky on the GPU when torch sees one, else on the CPU with one thread per actor.
That is exact: H, Sy, Kb and Sb are diagonal and the prior's surface block depends only on the component and a
scalar norm. AC26-BSC matched ISOFIT to float32 rounding on every pixel of an EnMAP, a PRISMA and an AVIRIS-NG scene.

Falls back to ISOFIT's own code, never to a different answer:
  - ATM_FAST_AL=0, or an initializer / iteration count other than apply_oe's (algebraic, 1): ISOFIT's function;
  - a configuration outside what was verified (non-diagonal Seps, model discrepancy, per-pixel heuristic prior,
    non-reflectance surface states, a surface without the multicomponent prior, no torch): ISOFIT's run_chunks for
    every chunk of that actor, logged once;
  - any error inside a chunk: ISOFIT's run_chunks for that chunk, logged. Outputs are written only at the end of
    a chunk, so a failed fast attempt leaves nothing behind.
ATM_FAST_AL_DEVICE=cpu keeps the solve off the GPU. Ray 2.59 leaves CUDA_VISIBLE_DEVICES alone for actors that ask
for no GPU, so every actor sees the device; each sizes its batches to its share of the free GPU memory.
"""
from __future__ import annotations

import importlib
import logging
import os

import numpy as np
from spectral.io import envi

from isofit import ray
from isofit.core.common import envi_header, eps
from isofit.core.fileio import write_bil_chunk
from isofit.core.geometry import Geometry
from isofit.inversion.inverse_simple import invert_algebraic

AL = importlib.import_module("isofit.utils.analytical_line")   # isofit.utils re-exports the function under this name
Base = AL.Worker.__ray_actor_class__
OUTSIDE_RET_CONST = -0.01                                      # invert_analytical's value outside the windows


class _Unsupported(Exception):
    """A configuration the batched solve was not verified for: use ISOFIT's run_chunks."""


def analytical_line(*args, **kwargs):
    """Drop-in for isofit.utils.analytical_line.analytical_line (same arguments)."""
    if (os.environ.get("ATM_FAST_AL", "1") == "0" or kwargs.get("initializer", "algebraic") != "algebraic"
            or kwargs.get("num_iter", 1) != 1):
        logging.info("fast_al: off, running ISOFIT's own analytical line")
        return AL.analytical_line(*args, **kwargs)
    n = kwargs.get("n_cores", -1)
    n = os.cpu_count() if n in (-1, None) else n
    stock, AL.Worker = AL.Worker, ray.remote(num_cpus=1)(type("FastWorkerN", (FastWorker,), {"n_actors": n}))
    try:
        return AL.analytical_line(*args, **kwargs)
    finally:
        AL.Worker = stock


class FastWorker(Base):
    n_actors = 1   # set per call by analytical_line(): the GPU memory budget is shared between the actors

    def __init__(self, *args):
        super().__init__(*args)
        self.fast_off = None
        try:
            self._prepare()
        except Exception as e:   # noqa: BLE001 - anything here means "use ISOFIT's path"
            self.fast_off = f"{type(e).__name__}: {e}"
            logging.warning(f"fast_al: actor uses ISOFIT's per-pixel solve ({self.fast_off})")

    def _prepare(self):
        fm, s = self.fm, self.fm.surface
        if self.per_pixel_heuristic_prior:
            raise _Unsupported("per_pixel_heuristic_prior")
        if self.n_non_rfl_bands:
            raise _Unsupported("non-reflectance surface states")
        if fm.model_discrepancy is not None:
            raise _Unsupported("model discrepancy")
        if not (hasattr(s, "Sa_inv_normalized") and hasattr(s, "component_covs")):
            raise _Unsupported(f"surface {type(s).__name__} has no multicomponent prior")
        if not np.array_equal(s.analytical_iv_idx, np.arange(len(fm.idx_surface))):
            raise _Unsupported("analytical_iv_idx is not every surface state")
        import torch
        dev = os.environ.get("ATM_FAST_AL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            torch.set_num_threads(1)
        self.torch, self.dev = torch, torch.device(dev)
        n = len(fm.idx_surface)
        # Surface block of fm.Sa's Sa_inv per component, before its 1/norm^2 factor (forward.Sa, multicomp.Sa).
        prior = np.stack([s.Sa_inv_normalized[c] / np.mean(np.diag(s.component_covs[c])) for c in range(len(s.component_covs))])
        self.prior = torch.from_numpy(prior).to(self.dev)
        self.col = torch.from_numpy(np.searchsorted(s.analytical_iv_idx, self.winidx)).to(self.dev)
        self.outside = np.ones(n, bool)
        self.outside[np.concatenate((self.winidx, fm.idx_surf_nonrfl))] = False
        if dev == "cuda":   # three n x n fp64 batches live at once (A, its factor, its inverse)
            budget = 0.5 * torch.cuda.mem_get_info(self.dev)[0] / max(1, self.n_actors)
            self.batch = int(max(64, min(4096, budget / (3 * 8 * n * n))))
        else:
            self.batch = 256
        self.diag_checked = False
        logging.info(f"fast_al: batched per-pixel solve on {dev}, {n} surface states, batches of {self.batch}")

    def run_chunks(self, line_breaks: tuple, fill_value: float = -9999.0) -> None:
        if self.fast_off:
            return super().run_chunks(line_breaks, fill_value)
        try:
            return self._run_fast(line_breaks)
        except _Unsupported as e:
            self.fast_off = str(e)
            logging.warning(f"fast_al: actor uses ISOFIT's per-pixel solve from here ({e})")
        except Exception as e:   # noqa: BLE001 - redo this chunk with ISOFIT's own code
            logging.exception(f"fast_al: chunk {line_breaks} failed ({e}); redoing it with ISOFIT's code")
        return super().run_chunks(line_breaks, fill_value)

    def _setup(self, r, c):
        """Worker.run_chunks before invert_analytical, unchanged (algebraic initializer, no heuristic prior)."""
        fm = self.fm
        meas = self.rdn[r, c, :]
        if self.radiance_correction is not None:
            meas = meas.copy() * self.radiance_correction
        if np.all(meas < 0):
            return None
        geom = Geometry(obs=self.obs[r, c, :], loc=self.loc[r, c, :], svf=self.svf[r, c] if len(self.svf) else 1,
                        bg_rfl=self.bg_rfl[r, c, :] if len(self.bg_rfl) else None, coszen=self.coszen,
                        full_config=self.config)
        x_atm = self.rt_state[r, c, :]
        iv_idx = fm.surface.analytical_iv_idx
        lbl = int(self.lbl[r, c, 0])
        sub = np.zeros(fm.nstate)
        sub[fm.idx_surface] = self.subs_state[lbl, 0, fm.full_idx][iv_idx]
        sub[fm.idx_atmosphere] = x_atm
        sub[fm.idx_instrument] = self.subs_state[lbl, 0, fm.full_idx][fm.idx_instrument]
        sub[np.isnan(sub)] = fm.init[np.isnan(sub)]
        xs, _, xi = fm.unpack(fm.init.copy())
        rfl_est, _ = invert_algebraic(fm, xs, x_atm, xi, meas, geom)
        x0 = np.concatenate([fm.surface.fit_params(rfl_est, geom), x_atm, xi])
        return meas, geom, fm.clip_bounds(x0, fm.bounds, eps=eps), sub

    def _check_diagonal(self, meas, geom, x):
        """Seps = Sy + Kb Sb Kb' must be diagonal. Its structure is set by the instrument model, so one pixel decides."""
        fm = self.fm
        Sy, Kb, Sb = fm.instrument.Sy(meas, geom), fm.Kb(x, geom), fm.Sb(meas)
        off = lambda M: M.shape[0] != M.shape[1] or np.count_nonzero(M - np.diag(np.diagonal(M)))
        if off(Sy) or off(Kb) or off(Sb):
            raise _Unsupported("Seps is not diagonal")
        self.diag_checked = True

    def _vectors(self, meas, geom, x, sub):
        """Everything invert_analytical needs for one pixel, as vectors (one iteration; al_batch2.vectors)."""
        fm, w = self.fm, self.winidx
        x_surface, x_atm, _ = fm.unpack(x)
        _, _, sub_instrument = fm.unpack(sub)
        _, rho_dif_dir = fm.calc_rfl(sub, geom)
        rho = fm.upsample(fm.surface.wl, rho_dif_dir)
        geom.bg_rfl = rho
        r, L_tot, L_dir_dir, L_dif_dir, L_dir_dif, L_dif_dif = fm.calc_atmosphere_quantities(x_atm, geom, rho_dif_dif=rho)
        L_atm, s_alb = fm.atmosphere.get_L_atm(x_atm, geom), r["sphalb"]
        L_bg = fm.calc_rdn_bgrfl(rho_dir_dif=rho, rho_dif_dif=rho, L_dir_dif=L_dir_dif, L_dif_dif=L_dif_dif,
                                 L_tot=L_tot, s_alb=s_alb)
        th = fm.surface.evaluate_theta(s_alb, geom, L_tot, L_dir_dir, L_dir_dif, L_dif_dir, L_dif_dif)[w]
        if not self.diag_checked:
            self._check_diagonal(meas, geom, x)
        Kb = fm.Kb(x, geom)
        seps = np.diagonal(fm.instrument.Sy(meas, geom))[w] + (np.diagonal(Kb) ** 2 * np.diagonal(fm.Sb(meas)))[w]
        s = fm.surface
        ci, nrm2 = s.component(x_surface, geom), s.norm(s.calc_lamb(x_surface, geom)[s.idx_ref]) ** 2
        y = meas[w] - L_atm[w] - fm.eof_offset(sub_instrument)[w] - L_bg[w]
        return x, ci, nrm2, th, seps, y, fm.xa(x, geom)[fm.idx_surface]

    def _solve(self, ci, nrm2, th, seps, y, xa):
        """A = prior(ci) / nrm2 + diag(th^2 / seps) on the window columns; returns the solution and diag(A^-1)."""
        torch, dev = self.torch, self.dev
        xk, var = np.empty(xa.shape), np.empty(xa.shape)
        for a in range(0, len(xa), self.batch):
            sl = slice(a, a + self.batch)
            d = lambda z: torch.from_numpy(np.ascontiguousarray(z[sl])).to(dev)
            A = self.prior[d(ci)] / d(nrm2)[:, None, None]
            b = torch.einsum("kij,kj->ki", A, d(xa))
            w, t = 1 / d(seps), d(th)
            A[:, self.col, self.col] += t ** 2 * w
            b[:, self.col] += t * d(y) * w
            L = torch.linalg.cholesky(A)
            xk[sl] = torch.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1).cpu().numpy()
            var[sl] = torch.diagonal(torch.cholesky_inverse(L), dim1=-2, dim2=-1).cpu().numpy()
        return xk, var

    def _finish(self, X, xk, var):
        """State rows with the solved surface -> rfl and unc rows (invert_analytical's tail + fill_statevector)."""
        fm = self.fm
        S = xk.copy()
        S[:, self.outside] = OUTSIDE_RET_CONST
        X = X.copy()
        X[:, fm.idx_surface] = S
        U = np.ones_like(X)
        U[:, fm.idx_surface] = np.sqrt(var)
        out = []
        for M in (X, U):
            F = np.empty((len(M), len(self.full_statevector)))
            F[:, fm.full_idx] = M
            F[:, fm.full_miss] = -9999.0
            out.append(F[:, self.full_idx_surf_rfl])
        return out

    def _run_fast(self, line_breaks: tuple) -> None:
        """Worker.run_chunks with the solve batched over the chunk; writes the same outputs the same way."""
        start, stop = line_breaks
        rfl = envi.open(envi_header(self.rfl_outpath)).open_memmap(interleave="bip", writable=False)[start:stop].copy()
        unc = envi.open(envi_header(self.unc_outpath)).open_memmap(interleave="bip", writable=False)[start:stop].copy()
        pairs = self.class_idx_pairs[(self.class_idx_pairs[:, 0] >= start) & (self.class_idx_pairs[:, 0] < stop)]
        rows, cols, V = [], [], []
        for r, c, *_ in pairs:
            p = self._setup(r, c)
            if p is not None:
                meas, geom, x0, sub = p
                V.append(self._vectors(meas, geom, x0, sub))
                rows.append(r - start)
                cols.append(c)
        if V:
            X, ci, nrm2, th, seps, y, xa = (np.array([v[i] for v in V]) for i in range(7))
            xk, var = self._solve(ci, nrm2, th, seps, y, xa)
            rfl[rows, cols], unc[rows, cols] = self._finish(X, xk, var)
        logging.info(f"Analytical line writing lines: {start} to {stop}. Surface: {self.surface_class_str} (fast_al)")
        shape = (self.n_lines, self.n_rfl_bands, self.n_samples)
        write_bil_chunk(np.swapaxes(rfl, 1, 2), self.rfl_outpath, start, shape)
        write_bil_chunk(np.swapaxes(unc, 1, 2), self.unc_outpath, start, shape)
