"""
Step 1 of the HyperSL anomaly bench: HyperSL's masked reconstruction of every valid pixel of the 4 labelled PRISMA
scenes, turned into anomaly score maps.

Runs in WSL Ubuntu on the laptop GPU (Windows torch fails here):
    ~/hypersl_eval/.venv/bin/python step1_hypersl_scores.py [set1 ...] [--limit N]

What HyperSL sees, exactly as it was trained (upstream mutihsi.py + train.py):
  - per-pixel spectra, no spatial context;
  - each band min-max scaled over the scene's valid pixels (sklearn minmax_scale per column);
  - wavelengths in nm, encoded by the model as (lambda - 400) / 10;
  - mask ratio 0.95: int(149 * 0.05) = 7 bands visible, 142 hidden.
The band axis is the 149-band PRISMA axis every earlier HyperSL experiment here used
(git show 4bbf91d:research/experiments/hypersl-splib-match/prisma_scene.py, drop_low_snr=True).

Reconstruction: R = 4 random masks per pixel (seeded); each band's reconstruction is the mean over the draws in
which it was hidden, so every residual is a prediction of a band the encoder did not see.

Score maps written to OUT/<set>_scores.npz (float32, 0 outside valid):
  hsl_l2    mean over bands of (x' - x_hat')^2, in min-max units - HyperSL's own reconstruction error
  hsl_loss  MSE + (1 - cosine) per pixel, in min-max units - its training objective, per pixel
  hsl_rm    the product's residual Mahalanobis (d 30, local window 21/61) on the residual in reflectance
            (x_hat' undone by the scene's own min and span) - the scorer Indradhanu is measured with
Validity and objects come from step 0 (the evaluator's own PrismaBench masks).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/mnt/c/Users/FITT/Documents/stage1_datasets")
OUT = ROOT / "hypersl_anomaly_bench"
MODEL_DIR = Path("/mnt/c/Users/FITT/Documents/hypersl_match_data/hypersl_model")   # ckpt.pt sha256 3d5f1914...a74a68c
REPO = Path("/mnt/c/Users/FITT/OneDrive/Documents/GitHub/hsi-anomaly-foundations-allotrope")
SCENES = {   # same files as app/evaluation/datasets.py PRISMA_SCENES
    "set1": "Set-1/Hypersepctral Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5",
    "set2": "Set-2/Hypersepctral Dataset/PRS_L2D_STD_20241205050514_20241205050518_0001.he5",
    "set3": "Set-3/Hyper/PRS_L2D_STD_20210516050459_20210516050503_0001.he5",
    "set4": "Set-4/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5",
}
SWATH = "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"
N_VNIR = 66
LOW_SNR_NM = ((400.0, 485.0), (1130.0, 1175.0), (1325.0, 1495.0), (1790.0, 2030.0), (2350.0, 2500.0))
OVERLAP_NM = (915.0, 985.0)
MASK_RATIO, DRAWS, BATCH = 0.95, 4, 4096


def load_rm():
    """The product's residual_mahalanobis module, loaded from its file (numpy only; avoids the app package)."""
    p = REPO / "app/utils/anomaly_detection/residual_mahalanobis.py"
    spec = importlib.util.spec_from_file_location("rm", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_cube(he5: Path) -> tuple[np.ndarray, np.ndarray]:
    """(149, H, W) float32 reflectance and (149,) ascending wavelengths, per-detector de-scaling."""
    with h5py.File(he5, "r") as f:
        a = f.attrs
        cw = np.concatenate([np.ravel(a["List_Cw_Vnir"]), np.ravel(a["List_Cw_Swir"])])
        flags = np.concatenate([np.ravel(a["List_Cw_Vnir_Flags"]), np.ravel(a["List_Cw_Swir_Flags"])])
        good = flags.astype(bool) & (cw > 0)
        idx = np.flatnonzero(good)[np.argsort(cw[good], kind="stable")]
        sel = cw[idx]
        drop = np.zeros(len(idx), bool)
        for lo, hi in LOW_SNR_NM:
            drop |= (sel >= lo) & (sel <= hi)
        drop |= (sel >= OVERLAP_NM[0]) & (sel <= OVERLAP_NM[1]) & (idx >= N_VNIR)
        idx = idx[~drop]
        b = {k: float(np.ravel(a[k])[0]) for k in ("L2ScaleVnirMin", "L2ScaleVnirMax", "L2ScaleSwirMin", "L2ScaleSwirMax")}
        vn = f[f"{SWATH}/VNIR_Cube"][...]                       # (rows, 66, cols) uint16, BIL
        sw = f[f"{SWATH}/SWIR_Cube"][...]                       # (rows, 173, cols)
    raw = np.concatenate([vn, sw], axis=1)                      # (rows, 239, cols)
    lo = np.r_[np.full(N_VNIR, b["L2ScaleVnirMin"]), np.full(raw.shape[1] - N_VNIR, b["L2ScaleSwirMin"])]
    hi = np.r_[np.full(N_VNIR, b["L2ScaleVnirMax"]), np.full(raw.shape[1] - N_VNIR, b["L2ScaleSwirMax"])]
    cube = np.empty((len(idx),) + (raw.shape[0], raw.shape[2]), np.float32)
    for j, i in enumerate(idx):
        cube[j] = lo[i] + raw[:, i, :].astype(np.float32) * (hi[i] - lo[i]) / 65535.0
    return cube, cw[idx].astype(np.float32)


def load_model(device: str):
    sys.path.insert(0, str(MODEL_DIR))
    from model import SpectralSharedEncoder               # upstream file, unmodified
    m = SpectralSharedEncoder(embedding_dim=256, encoder_depth=8, decoder_depth=4, num_heads=8)
    ck = torch.load(MODEL_DIR / "ckpt.pt", map_location="cpu", weights_only=False)
    m.load_state_dict({k.replace("module.", "", 1): v for k, v in ck["model"].items()})
    return m.eval().to(device), int(ck.get("epoch", -1)), sum(p.numel() for p in m.parameters())


@torch.no_grad()
def reconstruct(model, X: np.ndarray, wave: np.ndarray, device: str) -> np.ndarray:
    """(N, L) min-max spectra -> (N, L) mean reconstruction over the draws in which each band was hidden."""
    N, L = X.shape
    keep = int(L * (1 - MASK_RATIO))
    acc = np.zeros((N, L), np.float32)
    cnt = np.zeros((N, L), np.float32)
    last = np.zeros((N, L), np.float32)
    w_all = torch.tensor(wave, device=device).repeat(BATCH, 1)
    for r in range(DRAWS):
        torch.manual_seed(1000 + r)
        for i in range(0, N, BATCH):
            x = torch.from_numpy(X[i:i + BATCH]).to(device).unsqueeze(1)
            with torch.autocast("cuda", dtype=torch.float16):
                z, k, v, ids_restore, pos, _ = model.encoder_forward(x, w_all[:len(x)], MASK_RATIO)
                xh = model.decoder_forward(z, k, v, ids_restore, pos)
            xh = xh.float().squeeze(1)
            hid = (ids_restore >= keep).float()
            acc[i:i + BATCH] += (xh * hid).cpu().numpy()
            cnt[i:i + BATCH] += hid.cpu().numpy()
            last[i:i + BATCH] = xh.cpu().numpy()
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), last), int((cnt == 0).sum())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    names = [a for a in args if a in SCENES] or list(SCENES)
    device = "cuda"
    model, epoch, n_params = load_model(device)
    rm = load_rm()
    log = {"checkpoint_epoch": epoch, "parameters": n_params, "mask_ratio": MASK_RATIO, "draws": DRAWS,
           "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "scenes": {}}
    for name in names:
        t0 = time.time()
        cube, wave = read_cube(ROOT / SCENES[name])
        m = np.load(OUT / f"{name}_masks.npz")
        valid = m["valid"].astype(bool)
        assert valid.shape == cube.shape[1:], (valid.shape, cube.shape)
        rows, cols = np.nonzero(valid)
        if limit:
            rows, cols = rows[:limit], cols[:limit]
        X = cube[:, rows, cols].T                                  # (N, L) reflectance
        lo, hi = X.min(0), X.max(0)                                # the scene's own per-band frame
        span = np.maximum(hi - lo, 1e-9)
        Xs = ((X - lo) / span).astype(np.float32)
        t1 = time.time()
        Xh, never = reconstruct(model, Xs, wave, device)
        t_gpu = time.time() - t1
        r = Xs - Xh
        l2 = (r ** 2).mean(1)
        cos = (Xs * Xh).sum(1) / (np.linalg.norm(Xs, axis=1) * np.linalg.norm(Xh, axis=1) + 1e-12)
        loss = l2 + (1 - cos)
        out = {k: np.zeros(valid.shape, np.float32) for k in ("hsl_l2", "hsl_loss", "hsl_rm")}
        out["hsl_l2"][rows, cols], out["hsl_loss"][rows, cols] = l2, loss
        recon = np.zeros_like(cube)
        recon[:, rows, cols] = (Xh * span + lo).T
        keep = np.zeros(valid.shape, bool)
        keep[rows, cols] = True
        cube_k = np.where(keep[None], cube, 0).astype(np.float32)
        out["hsl_rm"] = rm.residual_mahalanobis(cube_k, recon, keep)
        tag = "" if not limit else f"_limit{limit}"
        np.savez_compressed(OUT / f"{name}_scores{tag}.npz", **out)
        log["scenes"][name] = {"bands": int(len(wave)), "pixels": int(len(rows)), "gpu_s": round(t_gpu, 1),
                               "total_s": round(time.time() - t0, 1), "bands_never_hidden": never,
                               "mean_l2": float(l2.mean()), "median_cos": float(np.median(cos)),
                               "wavelength_min_max": [float(wave.min()), float(wave.max())]}
        print(name, log["scenes"][name], flush=True)
        del cube, recon, cube_k, X, Xs, Xh
    (OUT / f"step1_log{'' if not limit else '_limit'}.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
