"""Numbers for chapter 21 (ch:crossstem). Run: C:/av/Scripts/python.exe compute_crossstem.py"""
import json, math, os, sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from app.foundation_models.components.spectral_attention import SpectralCrossAttentionCompressor

m = SpectralCrossAttentionCompressor()
print("compressor params", sum(p.numel() for p in m.parameters()))
print({n: p.numel() for n, p in m.named_parameters()})
print("d=16,D=32 params", sum(p.numel() for p in SpectralCrossAttentionCompressor(32, 16, 4).parameters()))
wl = torch.tensor([b["wavelength_nm"] for b in json.load(open(os.path.join(ROOT, "app/constants/spectralearth_bands.json")))["bands"]])
torch.manual_seed(0)
x = torch.randn(1, 202, 4, 4); v = torch.ones_like(x, dtype=torch.bool); v[:, 100] = False
x2 = x.clone(); x2[:, 100] = 99.0
with torch.no_grad():
    print("99-test masked max change", (m(x, wl, v) - m(x2, wl, v)).abs().max().item())
    print("99-test unmasked max change", (m(x, wl) - m(x2, wl)).abs().max().item())
for s in ([2., 1., 0., -1.], [2., 1., float("-inf"), -1.]):
    t = torch.tensor(s); print("softmax", s, [round(a, 4) for a in torch.softmax(t, 0).tolist()], "exps", [round(a, 4) for a in t.exp().tolist()])
MiB, GiB = 2 ** 20, 2 ** 30
macs = 2 * 202 * 1024 + 32 * 202 * 32 * 2 + 32 * 1024 + 32 * 32
print("MACs/pixel", macs, "ratio vs 5280", round(macs / 5280, 2), "vs 6464", round(macs / 6464, 2), "per patch", macs * 16384, "batch128", macs * 16384 * 128)
print("attn weights fp32 MiB", 16384 * 4 * 32 * 202 * 4 / MiB, "K+V fp32 MiB", 16384 * 2 * 202 * 32 * 4 / MiB,
      "self weights GiB", 16384 * 4 * 202 * 202 * 4 / GiB, "K+V 51 tok MiB", 16384 * 2 * 51 * 32 * 4 / MiB,
      "one (P,202,32) bf16 MiB", 16384 * 202 * 32 * 2 / MiB)
