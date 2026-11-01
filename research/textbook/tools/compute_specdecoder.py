"""Numbers for chapter 22 (ch:specdecoder). Run: C:/av/Scripts/python.exe compute_specdecoder.py"""
import json, os, sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from app.foundation_models.components.spectral_attention import SpectralCrossAttentionDecoder
from app.foundation_models.components.spectral_gate import SpectralGate
from app.foundation_models.components.spectral_reduced import SpectralReducedCompressor, WavelengthLinearDecoder

def n(m): return sum(p.numel() for p in m.parameters())
for name, m in [("decoder", SpectralCrossAttentionDecoder()), ("gate202", SpectralGate(202)), ("gate165", SpectralGate(165)),
                ("reduced", SpectralReducedCompressor()), ("wavelength_linear_decoder", WavelengthLinearDecoder())]:
    print(name, n(m), {k: p.numel() for k, p in m.named_parameters()})
wl = torch.tensor([b["wavelength_nm"] for b in json.load(open(os.path.join(ROOT, "app/constants/spectralearth_bands.json")))["bands"]])
r = SpectralReducedCompressor(); h = torch.bincount(r.bin_index(wl), minlength=r.n_bins)
print("bins", r.n_bins, "occupied", int((h > 0).sum()), "max", int(h.max()), "empty", [k for k in range(r.n_bins) if h[k] == 0])
print("counts", h.tolist())
torch.manual_seed(0)
x = torch.randn(1, 202, 4, 4); v = torch.ones_like(x, dtype=torch.bool); v[:, 100] = False
x2 = x.clone(); x2[:, 100] = 99.0
with torch.no_grad():
    print("reduced 99-test masked max change", (r(x, wl, v) - r(x2, wl, v)).abs().max().item())
red = 202 * 1696 + 2 * 53 * 1024 + 2 * 32 * 53 * 32 + 33792 + 2 * 202 * 32
per = 861184 + 1024 + 2 * 32 * 1024 + 3 * 202 * 32 * 32 + 202 * 32
print("MACs per-band", per, "reduced", red, "ratio", round(per / red, 2))
print("53-slot tensor bf16 MiB", 16384 * 53 * 32 * 2 / 2 ** 20)
