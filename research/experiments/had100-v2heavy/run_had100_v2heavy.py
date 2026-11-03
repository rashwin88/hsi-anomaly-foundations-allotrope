"""
Indradhanu V2 Heavy (v2.1.0 = FTH3 epoch 5) on HAD100's official protocol, the same harness as V1's
step6 (research/experiments/dual-window-mahalanobis/step6_had100_protocol.py): 100 crops of 64x64
(main.py crop_dict), the 276 good bands, then the first 50 / 100 / 200; mAUC = mean per-crop ROC-AUC.

Unlike V1, V2 reads any band list by wavelength, so each band setting is a separate V2 run fed exactly
those bands (minus any the checkpoint never trained on, SpectralStemInferencer.read_bands).

Input scaling (HAD100 is at-sensor radiance, V2 was trained on reflectance):
  A "pooled affine" (primary; V1's handling): each band mapped affinely onto the checkpoint's own
    training mean/std at that wavelength, ONE fit pooled over all 94 full tiles.
  B "blackbody TOA": rho = pi L / E_sun(lambda), E_sun a 5778 K Planck sun scaled to 1361 W/m^2
    (no irradiance table is available offline), cos(SZA) unknown so one scalar per tile matches
    the tile's mean to the training mean. Keeps the spectral shape of TOA reflectance.

Scorers (product code, app/utils/anomaly_detection/residual_mahalanobis.py): RM global mean (V1's
headline), RM local mean 5/15 (V1's variant), RM product default 21/61, and L1.
Crops are reflect-padded 32 px to 128 (the model's patch) and cropped back, as V1. All pixels scored.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DW = REPO / "research" / "experiments" / "dual-window-mahalanobis"
sys.path[:0] = [str(REPO), str(DW)]
import step3_public as p3  # noqa: E402
import step6_had100_protocol as p6  # noqa: E402

p3.BENCH = Path("C:/es/benchmarks/had100")
CKPT = Path("C:/es/models/FTH3_ep5.pt")
OUT = HERE / "results"
SETTINGS = {"50": 50, "100": 100, "200": 200, "full": 276}


def planck_sun(wl_nm: np.ndarray) -> np.ndarray:
    """Solar spectral irradiance at 1 AU, W m^-2 nm^-1, from a 5778 K blackbody scaled to 1361 W m^-2."""
    h, c, k, T = 6.626e-34, 2.998e8, 1.381e-23, 5778.0
    lam = np.linspace(100, 1e5, 200_000) * 1e-9
    B = lambda l: 2 * h * c ** 2 / l ** 5 / np.expm1(h * c / (l * k * T))  # noqa: E731
    return 1361.0 * B(wl_nm * 1e-9) / np.trapezoid(B(lam), lam) * 1e-9


def main() -> None:
    from app.foundation_models.inferencers.spectral_stem_inferencer import SpectralStemInferencer
    from app.models.training.inference_config import InferenceConfig
    from app.utils.anomaly_detection.residual_mahalanobis import residual_mahalanobis
    from app.utils.anomaly_detection.scoring import compute_score

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    s_wl, s_mu, s_sd = (sd[k].numpy().astype(np.float64) for k in ("stats.wl", "stats.mean", "stats.std"))
    inf = SpectralStemInferencer(InferenceConfig(
        foundation_model_name="spectral_stem_mae", model_config=ck["config"]["model_config"],
        checkpoint_path=str(CKPT), patch_size=128, stride=32, inference_batch_size=8))
    del ck, sd

    # pooled per-band radiance statistics over the 94 full tiles (variant A), per native band index
    n, s1, s2, wls = 0, 0.0, 0.0, []
    for _, c, wl, _ in p3.had100_scenes():
        X = np.nan_to_num(c).reshape(c.shape[0], -1).astype(np.float64)
        n, s1, s2 = n + X.shape[1], s1 + X.sum(1), s2 + (X ** 2).sum(1)
        wls.append(wl)
    l_mu = s1 / n
    l_sd = np.sqrt(np.maximum(s2 / n - l_mu ** 2, 0)) + 1e-9
    wl_spread = float(np.max(np.ptp(np.stack(wls), 0)))
    print(f"pooled over {len(wls)} tiles; max per-band wavelength spread across tiles {wl_spread:.2f} nm", flush=True)

    def scale(cube, wl, variant):
        t_mu, t_sd = np.interp(wl, s_wl, s_mu)[:, None, None], np.interp(wl, s_wl, s_sd)[:, None, None]
        if variant == "A":
            return (t_mu + t_sd * (cube - l_mu[:, None, None]) / l_sd[:, None, None]).astype(np.float32)
        rho = np.pi * cube * 1e-2 / planck_sun(wl)[:, None, None]        # uW cm^-2 sr^-1 nm^-1 -> W m^-2 ...
        return (rho * (t_mu.mean() / max(rho.mean(), 1e-9))).astype(np.float32)

    results, seg_report, t0 = {}, {}, time.time()
    for name, cube, wl, gt_full, x, y in p6.crops():
        cube = np.nan_to_num(cube)
        gt = gt_full[x:x + 64, y:y + 64]
        keep = np.ones(gt.shape, bool)
        res = results[name] = {"gt_px": int(gt.sum())}
        auc = lambda s: float(roc_auc_score(gt.ravel(), s.ravel()))  # noqa: E731
        for variant in ("A", "B"):
            full = scale(cube, wl, variant)
            for tag, nb in SETTINGS.items():
                idx = p6.GOOD_BANDS[:nb]
                sub, wsub, _ = inf.read_bands(full[idx][:, x:x + 64, y:y + 64], wl[idx])
                res[f"{variant} bands_read {tag}"] = int(len(wsub))
                pad = np.pad(sub, ((0, 0), (32, 32), (32, 32)), mode="reflect")
                with torch.no_grad():
                    rec = inf.predict_full_scene(torch.from_numpy(pad), torch.ones(1, 128, 128)).cpu().numpy()
                rec = rec[:, 32:96, 32:96].astype(np.float32)
                res[f"{variant} RM global {tag}"] = auc(residual_mahalanobis(sub, rec, keep, local_window=None))
                res[f"{variant} RM local 5/15 {tag}"] = auc(residual_mahalanobis(sub, rec, keep, local_window=(5, 15)))
                res[f"{variant} RM product 21/61 {tag}"] = auc(residual_mahalanobis(sub, rec, keep))
                res[f"{variant} L1 {tag}"] = auc(compute_score(sub, rec, keep, method="L1"))
                if tag == "full":   # what the product's thick-cloud exclusion WOULD remove (not applied)
                    cls, _ = inf.segment_full_scene(pad, np.ones((128, 128), bool), thin_cloud=False)
                    seg_report.setdefault(variant, {})[name] = {
                        k: int((cls[32:96, 32:96] == i).sum()) for i, k in enumerate(inf.CLASS_NAMES) if i}
        print(f"{len(results):3d} {name:28s} A RM {res['A RM global full']:.4f}  B RM {res['B RM global full']:.4f}"
              f"  {time.time() - t0:6.0f}s", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "had100_v2heavy_per_crop.json").write_text(json.dumps(results, indent=1))
    (OUT / "seg_head_classes_per_crop.json").write_text(json.dumps(seg_report, indent=1))
    (OUT / "runtime.json").write_text(json.dumps({"seconds": time.time() - t0, "crops": len(results),
                                                  "device": str(inf.device), "wl_spread_nm": wl_spread}))
    print("done", time.time() - t0)


if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)
    main()
