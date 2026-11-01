"""Numbers computed for chapter 37 (shipping Indradhanu V2). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_shipping.py > research/textbook/tools/compute_shipping_output.txt
"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")

# codename routing: sorted directories, dict keeps the last value
print("ord h s a", ord("h"), ord("s"), ord("a"))
dirs = sorted(["spectral_stem_finetune", "hyperspectral_segformer_mae"])
print("sorted", dirs, {"indradhanu": d for d in dirs})
dirs2 = sorted(["a_spectral_stem", "hyperspectral_segformer_mae"])
print("sorted2", dirs2, {"indradhanu": d for d in dirs2})

# np.allclose tolerance at the grid ends
for b in (460.0, 2450.0):
    print("allclose tol at", b, 1e-8 + 1e-5 * b)
from app.models.dataset.vendables import DEFAULT_COMMON_WAVELENGTH_GRID as G
G = np.asarray(G, float)
print("grid vs itself", np.allclose(G, G), "shifted 0.01 nm", np.allclose(G + 0.01, G),
      "shifted 0.001 nm", np.allclose(G + 0.001, G))

# trained_band_mask distances
rec = np.array([2395.0, 2405.0, 2440.0])
for b in (2400.0, 2452.0, 2470.0):
    print("band", b, "nearest distance", np.abs(rec - b).min(), np.abs(rec - b).min() <= 15)

# parameter count of the shipped checkpoint
try:
    import torch
    sd = torch.load("C:/es/ship/spectral_stem_finetune/spectral_stem_finetune_v2.0.0_epoch10.pt",
                    map_location="cpu", weights_only=False)["model_state_dict"]
    manifest = sum(v.numel() for k, v in sd.items() if not k.startswith(("stats.", "trained_wl")))
    buffers = sum(v.numel() for k, v in sd.items() if "running_" in k or "num_batches" in k)
    print("manifest count", manifest, "BatchNorm buffers", buffers, "parameters", manifest - buffers)
    groups = {}
    for k, v in sd.items():
        if k.startswith(("stats.", "trained_wl")) or "running_" in k or "num_batches" in k:
            continue
        groups[k.split(".")[0]] = groups.get(k.split(".")[0], 0) + v.numel()
    print("groups", groups, sum(groups.values()))
    print("excluded buffers", {k: v.numel() for k, v in sd.items() if k.startswith(("stats.", "trained_wl"))})
except Exception as e:
    print("checkpoint not read:", e)

# fit set changes every score
land = [0.02, -0.02, 0.02, -0.02]; cloud = [0.06, -0.06]; target = 0.08
v_all = sum(x * x for x in land + cloud) / 6
v_land = sum(x * x for x in land) / 4
print("spread all", math.sqrt(v_all), target / math.sqrt(v_all), "spread land", math.sqrt(v_land), target / math.sqrt(v_land))

# excluded_percent
print("excluded_percent", round(100 * 3 / 14, 3), round(100 * 4 / 26, 3))

# segmentation windows: padding and coverage
T, S = 128, 64
for n in (256, 300):
    pad = (-(n - T) % S) if n > T else T - n
    starts = list(range(0, n + pad - T + 1, S))
    print("n", n, "pad", pad, "starts", starts)
cov = np.zeros(320)
for y in range(0, 320 - T + 1, S):
    cov[y:y + T] += 1
print("coverage of row 100", cov[100], "row 10", cov[10], "row 300", cov[300])
print("mean cloud prob", (0.62 + 0.48 + 0.55 + 0.39) / 4)
