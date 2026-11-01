"""Numbers for chapter 23 (losses). Run from the repo root with a Python that has torch:
    python research/textbook/tools/compute_23-losses.py > research/textbook/tools/compute_23-losses_output.txt
Every number printed here appears in chapters/23-losses.tex or answers/23-losses.tex."""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.foundation_models.components.sam_loss import SAMLoss  # the real loss

print("== masked L1, 3 bands x 2 graded pixels (+1 visible pixel) ==")
x = torch.tensor([[0.100, 0.200, 0.300], [0.050, 0.100, 0.080], [0.20, 0.20, 0.20]])
xh = torch.tensor([[0.120, 0.180, 0.330], [0.060, 0.070, 0.080], [0.50, 0.50, 0.50]], requires_grad=True)
m = torch.tensor([1.0, 1.0, 0.0])
per_pix = (xh - x).abs().mean(dim=1)
l1 = per_pix[m == 1].mean()
l1.backward()
print("per-pixel L1:", per_pix.detach().numpy())
print("L1:", float(l1))
print("grad dL1/dxhat:\n", xh.grad.numpy())

print("== L1 vs L2 minimisers ==")
y = np.array([0.10, 0.11, 0.12, 0.13, 0.90])
print("mean", y.mean(), "median", np.median(y))
for c in (0.12, 0.272):
    print(f"c={c}: sum|y-c|={np.abs(y - c).sum():.4f} sum(y-c)^2={((y - c) ** 2).sum():.6f}")
print("L2 grad terms at c=0.12:", 2 * (0.12 - y))
print("L1 grad terms at c=0.12 (sign):", np.sign(0.12 - y))

print("== SAM worked example ==")
a = np.array([0.1, 0.2, 0.3]); b = np.array([0.12, 0.18, 0.33])
dot = a @ b; na = np.linalg.norm(a); nb = np.linalg.norm(b); u = dot / (na * nb)
print("dot", dot, "|x|", na, "|xhat|", nb, "prod", na * nb, "u", u, "theta", math.acos(u), "deg", math.degrees(math.acos(u)))
print("1-u^2", 1 - u * u, "sqrt", math.sqrt(1 - u * u), "darccos", -1 / math.sqrt(1 - u * u))
du = b * 0  # placeholder
du = (a / (na * nb)) - dot * b / (na * nb ** 3)
dth = -1 / math.sqrt(1 - u * u) * du
print("du/dxhat", du, "dtheta/dxhat", dth, "|dtheta|", np.linalg.norm(dth), "1/|xhat|", 1 / nb)
t = torch.tensor(b, dtype=torch.float64, requires_grad=True)
th = torch.acos((torch.tensor(a, dtype=torch.float64) @ t) / (torch.tensor(a, dtype=torch.float64).norm() * t.norm()))
th.backward(); print("autograd dtheta/dxhat (float64 arccos):", t.grad.numpy())
xh_t = torch.tensor(b, dtype=torch.float32).reshape(1, 3, 1, 1).requires_grad_(True)
x_t = torch.tensor(a, dtype=torch.float32).reshape(1, 3, 1, 1)
loss = SAMLoss()(xh_t, x_t, torch.ones(1, 1, 1, 1)); loss.backward()
print("SAMLoss value", float(loss), "grad", xh_t.grad.flatten().numpy())
s2 = (na * nb) ** 2 - dot ** 2
print("atan2 parts: cross", na * nb, "cross^2", (na * nb) ** 2, "dot^2", dot ** 2, "sin_sq", s2, "sin", math.sqrt(s2), "theta", math.atan2(math.sqrt(s2), dot))

print("== contrast: xhat = 2x ==")
print("L1", np.abs(2 * a - a).mean(), "u", (a @ (2 * a)) / (na * np.linalg.norm(2 * a)))

print("== arccos derivative table ==")
for uu in (0.0, 0.5, 0.9, 0.99, 0.999, 0.9999):
    print(uu, "1-u^2", 1 - uu * uu, "sqrt", math.sqrt(1 - uu * uu), "deriv", -1 / math.sqrt(1 - uu * uu), "theta", math.acos(uu), "deg", math.degrees(math.acos(uu)))

print("== float32 cos resolution near 1 ==")
sp = float(np.spacing(np.float32(1.0) - np.float32(1e-7)))
print("float32 spacing just below 1:", 2.0 ** -24, "np", np.float32(1) - np.nextafter(np.float32(1), np.float32(0)))
for th_ in (1e-3, 5e-4, 3e-4, 2e-4, 1e-4):
    c32 = np.float32(math.cos(th_))
    print(th_, "cos in float32:", repr(c32), "== 1?", c32 == np.float32(1))
xh1 = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
x1 = torch.tensor([0.1, 0.2, 0.3])
uu = (x1 @ xh1) / (x1.norm() * xh1.norm())
th1 = torch.acos(uu.clamp(-1, 1)); th1.backward()
print("naive arccos at perfect reconstruction, float32: u", float(uu), "theta", float(th1), "grad", xh1.grad.numpy())
xh2 = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
th2 = torch.acos(((x1 @ xh2) / (x1.norm() * xh2.norm()))); th2.backward()
print("naive arccos no clamp: theta", float(th2), "grad", xh2.grad.numpy())
xh3 = torch.tensor([0.1, 0.2, 0.3]).reshape(1, 3, 1, 1).requires_grad_(True)
l3 = SAMLoss()(xh3, x1.reshape(1, 3, 1, 1), torch.ones(1, 1, 1, 1)); l3.backward()
print("SAMLoss at perfect reconstruction: theta", float(l3), "grad", xh3.grad.flatten().numpy())
print("eps floor angle 3-band: atan(1e-4/0.14) =", math.atan2(1e-4, 0.14))
print("eps floor angle 165 bands at 0.2: |x|^2 =", 165 * 0.04, "angle", math.atan2(1e-4, 165 * 0.04))

print("== ramp ==")
for R, lm in ((10, 0.5), (20, 1.0)):
    print(R, lm, [round(lm * min(1.0, t / R), 4) for t in range(0, 22)])
print("combined epoch-100:", 0.010 + 0.5 * 0.060, "shares", 0.010 / 0.040, 0.030 / 0.040)
print("ramp artefact: e0", 0.090 + 0 * 0.35, "e10", 0.040 + 0.5 * 0.20)

print("== naive arccos vs SAMLoss when xhat == x, float32 ==")
for v in ([0.2, 0.2, 0.2], [0.25, 0.5, 0.125]):
    xv = torch.tensor(v)
    a1 = xv.clone().requires_grad_(True)
    u1 = (xv @ a1) / (xv.norm() * a1.norm()); t1 = torch.acos(u1); t1.backward()
    a2 = xv.clone().requires_grad_(True)
    u2 = (xv @ a2) / (xv.norm() * a2.norm()); t2 = torch.acos(u2.clamp(-1, 1)); t2.backward()
    a3 = xv.clone().reshape(1, 3, 1, 1).requires_grad_(True)
    t3 = SAMLoss()(a3, xv.reshape(1, 3, 1, 1), torch.ones(1, 1, 1, 1)); t3.backward()
    print(v, "u =", repr(float(u1)), "| arccos:", float(t1), a1.grad.numpy(),
          "| clamped arccos:", float(t2), a2.grad.numpy(), "| SAMLoss:", float(t3), a3.grad.flatten().numpy())

print("== SAM gradient intermediates (float64) ==")
a = np.array([0.1, 0.2, 0.3]); b = np.array([0.12, 0.18, 0.33])
s = a @ b; na = np.linalg.norm(a); nb = np.linalg.norm(b)
print("a*b", na * nb, "b^3", nb ** 3, "a*b^3", na * nb ** 3, "s/(a b^3)", s / (na * nb ** 3))
print("x/(ab)", a / (na * nb), "s xhat/(a b^3)", s * b / (na * nb ** 3))
du = a / (na * nb) - s * b / (na * nb ** 3)
print("du", du, "xhat.du", b @ du, "|du|", np.linalg.norm(du), "sqrt(1-u^2)/|xhat|", math.sqrt(1 - (s / (na * nb)) ** 2) / nb)
print("unit x", a / na, "unit xhat", b / nb)
print("chain rule demo f(w)=(3w+1)^2 at w=1:", (3 * 1 + 1) ** 2, "deriv", 2 * (3 * 1 + 1) * 3, "numeric", ((3 * 1.001 + 1) ** 2 - (3 * 0.999 + 1) ** 2) / 0.002)
