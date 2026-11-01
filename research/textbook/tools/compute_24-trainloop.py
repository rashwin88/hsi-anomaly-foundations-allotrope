"""Numbers for chapter 24 (training loop). Run from the repo root with a Python that has torch:
    python research/textbook/tools/compute_24-trainloop.py > research/textbook/tools/compute_24-trainloop_output.txt
Every number printed here appears in chapters/24-trainloop.tex or answers/24-trainloop.tex."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.models.patches.patching_request import PatchRequest
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator

L = lambda w: (w - 3.0) ** 2
G = lambda w: 2.0 * (w - 3.0)

print("== plain gradient descent, L(w)=(w-3)^2, w0=0 ==")
for eta in (0.1, 1.1):
    w = 0.0
    for t in range(1, 4):
        g = G(w); w = w - eta * g
        print(f"eta={eta} step {t}: g={g:.6f} w={w:.6f} L={L(w):.6f}")

print("== momentum beta=0.9, eta=0.1 (PyTorch SGD form v=beta v + g) ==")
w, v = 0.0, 0.0
for t in range(1, 4):
    g = G(w); v = 0.9 * v + g; w = w - 0.1 * v
    print(f"step {t}: g={g:.6f} v={v:.6f} w={w:.6f} L={L(w):.6f}")
p = torch.nn.Parameter(torch.tensor(0.0)); opt = torch.optim.SGD([p], lr=0.1, momentum=0.9)
for t in range(3):
    opt.zero_grad(); L(p).backward(); opt.step()
print("torch SGD momentum w after 3:", float(p))

print("== bias-correction factors ==")
for t in (1, 2, 3, 10, 100, 1000):
    print(t, "1-0.9^t", 1 - 0.9 ** t, "1-0.999^t", 1 - 0.999 ** t)

print("== Adam by hand, eta=0.1 ==")
w, m, v = 0.0, 0.0, 0.0
for t in range(1, 4):
    g = G(w)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    mh = m / (1 - 0.9 ** t); vh = v / (1 - 0.999 ** t)
    step = 0.1 * mh / (math.sqrt(vh) + 1e-8)
    w = w - step
    print(f"t={t}: g={g:.6f} m={m:.6f} v={v:.6f} mhat={mh:.6f} vhat={vh:.6f} sqrt={math.sqrt(vh):.6f} step={step:.6f} w={w:.6f}")
p = torch.nn.Parameter(torch.tensor(0.0)); opt = torch.optim.Adam([p], lr=0.1)
for t in range(3):
    opt.zero_grad(); L(p).backward(); opt.step(); print("torch Adam w:", float(p))
print("defaults:", torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))]).defaults)

print("== Adam first step is eta whatever the gradient scale ==")
for gs in (1e-4, 1.0, 1e4):
    m = 0.1 * gs; v = 0.001 * gs * gs
    print(gs, "step", 0.1 * (m / 0.1) / (math.sqrt(v / 0.001) + 1e-8))

print("== L2-in-Adam vs AdamW: two weights w=1, lam=0.01, gradient (loss part) zero, sqrt(vhat) given ==")
lam, eta = 0.01, 1e-3
for name, sv in (("A (big gradients)", 10.0), ("B (small gradients)", 0.01)):
    coupled = eta * (lam * 1.0) / sv          # decay passed through Adam's normaliser
    decoupled = eta * lam * 1.0               # AdamW
    print(name, "coupled shrink", coupled, "decoupled shrink", decoupled)

print("== cosine annealing, the real scheduler ==")
def schedule(lr0, T, eta_min):
    p = torch.nn.Parameter(torch.zeros(1)); o = torch.optim.Adam([p], lr=lr0)
    s = torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=T, eta_min=eta_min)
    out = []
    for e in range(T + 1):
        out.append(o.param_groups[0]["lr"])
        o.step(); s.step()
    return out
v2 = schedule(1e-3, 200, 1e-6)
closed = lambda t: 1e-6 + 0.5 * (1e-3 - 1e-6) * (1 + math.cos(math.pi * t / 200))
for t in (0, 1, 2, 25, 50, 100, 101, 150, 175, 199, 200):
    print(f"v0.2.0 t={t}: torch {v2[t]:.6e} closed {closed(t):.6e} cos={math.cos(math.pi*t/200):.6f}")
f1 = schedule(5e-4, 200, 1e-6)
print("F001 t=0,100,199:", f1[0], f1[100], f1[199])
with open(Path(__file__).with_name("compute_24-trainloop_cosine.dat"), "w") as fh:
    fh.write("t v02 f001\n")
    for t in range(0, 201, 2):
        fh.write(f"{t} {v2[t]*1e3:.6f} {f1[t]*1e3:.6f}\n")
print("warmup example (W=5, from 1e-6 to 1e-3):", [1e-6 + (1e-3 - 1e-6) * t / 5 for t in range(6)])

print("== gradient clipping by norm ==")
q = torch.nn.Parameter(torch.zeros(3)); q.grad = torch.tensor([3.0, 4.0, 12.0])
n = torch.nn.utils.clip_grad_norm_([q], max_norm=1.0)
print("norm before", float(n), "clipped", q.grad.numpy(), "norm after", float(q.grad.norm()))

print("== bf16 resolution ==")
for val in (0.2, 0.2003, 0.2005, 0.0300, 0.0301):
    print(val, "bf16 ->", float(torch.tensor(val).to(torch.bfloat16)), "fp32 ->", float(torch.tensor(val)))
print("bf16 spacing at 0.2:", 2.0 ** -3 * 2.0 ** -7, "at 0.03:", 2.0 ** -6 * 2.0 ** -7)

print("== bytes ==")
print("patch bytes", 165 * 128 * 128 * 4, "batch of 128", 165 * 128 * 128 * 4 * 128)
print("presentations v0.2.0", 40000 * 200, "steps/epoch", 40000 / 128, math.ceil(40000 / 128))
print("F001 effective batch", 8 * 4, "optimizer steps/epoch", 30000 / 32)

print("== patch plans (real PatchPlanGenerator) ==")
for H in (300, 500, 256, 128):
    req = PatchRequest(input_cube=(165, H, 128), width=128, height=128, stride=64)
    plan = PatchPlanGenerator().generate_patching_plan(req)
    print(H, sorted({r for r, c in plan.patch_coordinates}), [r for r, c in plan.patch_coordinates])

print("== sample-weighted average ==")
print((0.05 * 3 + 0.03 * 128) / 131, (0.05 + 0.03) / 2)

print("== exercises ==")
w = 0.0
for t in range(3):
    g = G(w); w -= 0.25 * g; print("eta=0.25 step", t + 1, "g", g, "w", w)
m = v = 0.0
for t in (1, 2):
    m = 0.9 * m + 0.1 * 2.0; v = 0.999 * v + 0.001 * 4.0
    print("const g=2 t", t, "m", m, "v", v, "mhat", m / (1 - 0.9 ** t), "vhat", v / (1 - 0.999 ** t))
for t in (25, 175):
    print("cos lr", t, closed(t))
q = torch.nn.Parameter(torch.zeros(3)); q.grad = torch.tensor([1.0, 2.0, 2.0])
torch.nn.utils.clip_grad_norm_([q], max_norm=1.5); print("clip (1,2,2) at 1.5:", q.grad.numpy())
for H in (400,):
    req = PatchRequest(input_cube=(165, H, 300), width=128, height=128, stride=64)
    plan = PatchPlanGenerator().generate_patching_plan(req)
    print(H, "x 300 rows", sorted({r for r, c in plan.patch_coordinates}), "cols", sorted({c for r, c in plan.patch_coordinates}), "n", len(plan.patch_coordinates))
