"""Numbers for chapter 28 (inference). Run from the repo root:
    C:/av/Scripts/python.exe research/textbook/tools/compute_inference.py > research/textbook/tools/compute_inference_output.txt
Uses the product's own PatchPlanGenerator and checkerboard so the counts are the code's, not a re-derivation."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.models.patches.patching_request import PatchRequest
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

P, S = 128, 32


def axis_coords(n):
    """Row starts of the plan for an n x n scene (the column starts are the same list)."""
    plan = PatchPlanGenerator().generate_patching_plan(PatchRequest(input_cube=(1, n, n), width=P, height=P, stride=S))
    coords = plan.patch_coordinates
    k = math.isqrt(len(coords))
    assert k * k == len(coords)
    return [coords[i * k][0] for i in range(k)]


print("== tiles per axis: code vs ceil((W-P)/S)+1 ==")
for n in (128, 160, 600, 1000, 1024, 1205, 1210, 1212, 1219, 1200, 900):
    c = axis_coords(n)
    print(f"W={n:5d}  code={len(c):3d}  unique={len(set(c)):3d}  ceil+1={math.ceil((n - P) / S) + 1:3d}  "
          f"(W-P)/S={(n - P) / S:.4f}  first={c[:3]} last={c[-3:]}")

print("\n== whole-scene tiles and forward passes (batch 8, two passes per batch) ==")
for h, w in ((1000, 1000), (1210, 1219), (1205, 1212), (600, 600), (1200, 900)):
    t = len(axis_coords(h)) * len(axis_coords(w))
    b = math.ceil(t / 8)
    print(f"{h}x{w}: tiles={t} batches={b} forward_passes={2 * b}")

print("\n== coverage along one axis (W=1000): how many tiles contain each column ==")
c = axis_coords(1000)
cov = np.zeros(1000, int)
for s in c:
    cov[s:s + P] += 1
runs = []
start = 0
for x in range(1, 1001):
    if x == 1000 or cov[x] != cov[start]:
        runs.append((start, x - 1, int(cov[start])))
        start = x
for a, b, k in runs:
    print(f"columns {a:4d}-{b:4d}: {k} tiles")
print("2-D interior coverage = 4 x 4 =", 4 * 4)

print("\n== coverage along one axis (W=1219) ==")
c = axis_coords(1219)
cov = np.zeros(1219, int)
for s in c:
    cov[s:s + P] += 1
start = 0
for x in range(1, 1220):
    if x == 1219 or cov[x] != cov[start]:
        print(f"columns {start:4d}-{x - 1:4d}: {int(cov[start])} tiles")
        start = x

print("\n== memory of one (165, 1000, 1000) float32 cube ==")
print(165 * 1000 * 1000 * 4, "bytes")
print("(165, 1210, 1219):", 165 * 1210 * 1219 * 4, "bytes")

print("\n== checkerboard: hidden pattern vs selection pattern ==")
def checker(n, cell, invert=False):
    r = np.arange(n) // cell
    g = (r[:, None] + r[None, :]) % 2      # 1 = visible, 0 = hidden
    return 1 - g if invert else g
for block in (1, 2):
    hid1 = 1 - checker(4, block)           # hidden in pass 1 (keep mask uses mask_block)
    hid2 = 1 - checker(4, block, True)
    sel1 = 1 - checker(4, 1)               # infer() selects pass-1 values with checkerboard_cell_size = 1
    sel2 = checker(4, 1)
    leaked = (sel1 * (1 - hid1) + sel2 * (1 - hid2))
    print(f"mask_block={block}: hidden pass1=\n{hid1}\n selected from pass1=\n{sel1}\n tokens taken from a pass where VISIBLE: {int(leaked.sum())} of 16")
