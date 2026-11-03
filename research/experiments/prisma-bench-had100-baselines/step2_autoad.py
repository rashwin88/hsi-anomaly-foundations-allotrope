"""
Step 2: Auto-AD (Wang et al., IEEE TGRS 2021) on the 4 labelled PRISMA scenes, with the official network
(https://github.com/RSIDEA-WHU2020/Auto-AD, `models/skip.py`, `utils/common_utils.py`, imported unmodified from a
clone passed as --repo) and the official training loop of `AD_main_loop.py` re-typed here with its settings unchanged:
skip net 5 x 128 channels, reflection pad, LeakyReLU, sigmoid output, uniform noise input (var 0.1) + 0.1 regularising
noise each step, Adam lr 0.01, up to 1001 iterations, the adaptive weighting block every 100 iterations, early stop when
the mean |loss change| over 50 iterations < 1.5e-5. Score = the residual (sum over bands of squared error) from the last
weighting block, as the official code saves it.

Changes forced by the data, not by choice of result:
  - input: the evaluator's cube (PrismaBench native bands, reflectance), min-max scaled to [0, 1] over the scene's
    valid pixels (the sigmoid output needs [0, 1]; the official HYDICE .mat is [0, 1]);
  - tiling: the scene is cut into tiles of at most --tile px (default 2 x 2) and each tile is a separate Auto-AD run
    (one image per run, as the method is defined); scores are stitched back without re-normalisation.

    C:/av/Scripts/python.exe -X utf8 step2_autoad.py --repo <Auto-AD clone> [--scenes set1,...] [--tile 640]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from app.evaluation.datasets import PrismaBench  # noqa: E402

ROOT = Path(r"C:\Users\FITT\Documents\stage1_datasets")
OUT = ROOT / "had100_baselines"


def load_pkg(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path / "__init__.py", submodule_search_locations=[str(path)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def autoad_tile(img_np: np.ndarray, skip, get_noise, get_params, seed: int) -> tuple[np.ndarray, int]:
    """The official main() loop on one (B, h, w) image in [0, 1]; returns (residual (h, w), iterations run)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.cuda.FloatTensor
    thres, channellss, layers = 0.000015, 128, 5
    img_var = torch.from_numpy(img_np).type(dtype)
    band, row, col = img_var.size()
    LR, num_iter, reg_noise_std = 0.01, 1001, 0.1
    net = skip(band, band, num_channels_down=[channellss] * layers, num_channels_up=[channellss] * layers,
               num_channels_skip=[channellss] * layers, filter_size_up=3, filter_size_down=3, upsample_mode='nearest',
               filter_skip_size=1, need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU').type(dtype)
    net_input = get_noise(band, '2D', img_np.shape[1:]).type(dtype)
    mse = torch.nn.MSELoss().type(dtype)
    img_var = img_var[None, :].cuda()
    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    def closure(iter_num, mask_varr, residual_varr):
        net_in = net_input_saved + (noise.normal_() * reg_noise_std)
        out = net(net_in)
        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()
        if iter_num % 100 == 0 and iter_num != 0:
            temp = (out.detach()[0] - img_var[0]) ** 2
            residual_img = temp.sum(0)
            residual_var_clone = residual_img
            residual_img = residual_img.max() - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)
            mask_var_clone[0, :] = residual_img[None]
        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()
        return mask_var_clone, residual_var_clone, total_loss.item()

    optimizer = torch.optim.Adam(get_params('net', net, net_input), lr=LR)
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, loss = closure(j, mask_var, residual_varr)
        optimizer.step()
        end_iter = False
        if j >= 1:
            loss_np[0][j - int(j / 50) * 50 - 1] = abs(loss - loss_last)
            if j % 50 == 0 and np.mean(loss_np) < thres:
                end_iter = True
        loss_last = loss
        if j == num_iter - 1 or end_iter:
            return residual_varr.detach().cpu().numpy(), j + 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--scenes", default="set1,set2,set3,set4")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--crop", type=int, default=0)
    a = p.parse_args()
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    repo = Path(a.repo)
    models = load_pkg("autoad_models", repo / "models")
    spec = importlib.util.spec_from_file_location("autoad_common_utils", repo / "utils" / "common_utils.py")
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)
    out = OUT / "_crop" if a.crop else OUT
    tpath = out / "timing_autoad.json"
    timing = json.loads(tpath.read_text()) if tpath.exists() else {}
    for scene in PrismaBench(str(ROOT), a.scenes.split(",")):
        d = out / scene.name
        d.mkdir(parents=True, exist_ok=True)
        if (d / "AutoAD.npz").exists():
            continue
        X, valid = scene.cube, scene.valid
        if a.crop:
            H, W = valid.shape
            sl = (slice(H // 2 - a.crop // 2, H // 2 + a.crop // 2), slice(W // 2 - a.crop // 2, W // 2 + a.crop // 2))
            X, valid = np.ascontiguousarray(X[(slice(None),) + sl]), valid[sl].copy()
        lo, hi = X[:, valid].min(), X[:, valid].max()
        Xn = np.clip((X - lo) / (hi - lo), 0, 1).astype(np.float32)
        C, H, W = Xn.shape
        ny, nx = -(-H // a.tile), -(-W // a.tile)
        ys, xs = np.linspace(0, H, ny + 1).astype(int), np.linspace(0, W, nx + 1).astype(int)
        score = np.zeros((H, W), np.float32)
        t0, iters = time.time(), []
        for i in range(ny):
            for k in range(nx):
                tile = np.ascontiguousarray(Xn[:, ys[i]:ys[i + 1], xs[k]:xs[k + 1]])
                res, n = autoad_tile(tile, models.skip, cu.get_noise, cu.get_params, seed=1000 * i + k)
                score[ys[i]:ys[i + 1], xs[k]:xs[k + 1]] = res
                iters.append(n)
                print(scene.name, "tile", i, k, tile.shape, "iters", n, round(time.time() - t0, 1), flush=True)
                torch.cuda.empty_cache()
        np.savez_compressed(d / "AutoAD.npz", score=score * valid)
        timing[scene.name] = {"s": round(time.time() - t0, 1), "tiles": ny * nx, "iters": iters,
                              "minmax": [float(lo), float(hi)]}
        tpath.write_text(json.dumps(timing, indent=1))


if __name__ == "__main__":
    main()
