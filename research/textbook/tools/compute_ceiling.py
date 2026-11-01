"""Numbers for chapter 20 (ch:ceiling). Run: C:/av/Scripts/python.exe compute_ceiling.py
Uses the repo's own wavelength_encoding (app/foundation_models/components/spectral_tokens.py)."""
import math, os, sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from app.foundation_models.components.spectral_tokens import wavelength_encoding

P = [10 * 1000 ** (i / 15) for i in range(16)]
print("periods (nm):", [round(p, 2) for p in P])
for lam in (660, 670, 1650):
    print(lam, "sin/cos per period:", " ".join(f"{math.sin(2*math.pi*lam/p):+.4f}/{math.cos(2*math.pi*lam/p):+.4f}" for p in P))
for dl in (10, 990):
    c = [math.cos(2 * math.pi * dl / p) for p in P]
    print("delta", dl, "cos terms", [round(x, 4) for x in c], "sum", round(sum(c), 4), "cos-sim", round(sum(c) / 16, 4))
e = wavelength_encoding(torch.tensor([660., 670., 1650.]))
cs = torch.nn.functional.cosine_similarity
print("module cos-sim 660/670", round(cs(e[0], e[1], dim=0).item(), 4), "660/1650", round(cs(e[0], e[2], dim=0).item(), 4))
print("module norms", [round(v, 4) for v in e.norm(dim=1).tolist()])
# cross-attention worked example (identity projections)
q = torch.tensor([1., 0.]); K = torch.tensor([[2., 0.], [0., 1.], [1., 1.]]); V = torch.tensor([10., 20., 30.])
w = torch.softmax(K @ q / math.sqrt(2), 0); print("xattn weights", [round(x, 4) for x in w.tolist()], "answer", round((w @ V).item(), 4))
q2 = torch.tensor([0., 1.]); w2 = torch.softmax(K @ q2 / math.sqrt(2), 0)
print("xattn q=[0,1] weights", [round(x, 4) for x in w2.tolist()], "answer", round((w2 @ V).item(), 4))
