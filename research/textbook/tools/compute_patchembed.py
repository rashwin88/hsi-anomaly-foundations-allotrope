"""Numbers for chapter 12-patchembed. Run from the repo root with the project venv:
    PYTHONPATH=. python research/textbook/tools/compute_patchembed.py > research/textbook/tools/compute_patchembed_output.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.foundation_models.components.hyperspectral_seg_former_mae import HyperspectralSegFormerMAE
from app.foundation_models.components.overlap_patch_embedding import OverlapPatchEmbedding

m = HyperspectralSegFormerMAE(compressed_channels=32, pixel_mean=[0.0] * 165, pixel_std=[1.0] * 165)
tot = 0
for i, pe in enumerate(m.encoder.patch_embeds):
    conv = sum(p.numel() for p in pe.proj.parameters())
    ln = sum(p.numel() for p in pe.norm.parameters())
    tot += conv + ln
    print(f"stage {i + 1}: {pe.proj}  conv {conv}  LayerNorm {ln}  total {conv + ln}")
print("all four patch embeddings:", tot, " fraction of model:", tot / sum(p.numel() for p in m.parameters()))
print("stage-final LayerNorms:", [sum(p.numel() for p in n.parameters()) for n in m.encoder.norms])
print("stage 1 without compressor (165->32, k4) conv + LN:",
      sum(p.numel() for p in OverlapPatchEmbedding(165, 4, 4, 32, padding=0).parameters()))
print("attention pairs, pixel tokens vs stage-1 tokens:", (128 * 128) ** 2, 1024 ** 2)

# Tiny patch embedding: 1 channel, 4x4 image, kernel 2, stride 2, two output channels.
img = torch.tensor([[1., 2., 0., 1.], [3., 4., 1., 0.], [0., 0., 2., 2.], [1., 1., 2., 2.]]).view(1, 1, 4, 4)
K = torch.tensor([[[[.25, .25], [.25, .25]]], [[[1., -1.], [1., -1.]]]])
tok = F.conv2d(img, K, stride=2).flatten(2).transpose(1, 2)
print("tiny tokens (B, N, C):", tok.tolist())

# LayerNorm on a 4-number token, and on the two-feature tokens.
ln = nn.LayerNorm(4)
print("LN([2,4,4,6]):", [round(v, 4) for v in ln(torch.tensor([[2., 4., 4., 6.]])).flatten().tolist()])
with torch.no_grad():
    ln.weight.copy_(torch.tensor([1., 1., 2., 2.])); ln.bias.copy_(torch.tensor([0., 0., 0., 1.]))
print("LN with gamma=(1,1,2,2), beta=(0,0,0,1):", [round(v, 4) for v in ln(torch.tensor([[2., 4., 4., 6.]])).flatten().tolist()])
ln2 = nn.LayerNorm(2)
print("LN on tiny tokens:", [[round(v, 5) for v in r] for r in ln2(tok[0]).tolist()])

# Coverage counts in 1-D: how many windows cover each pixel.
def cover(k, s, p, n):
    cnt = [0] * n
    i = 0
    while i * s - p + k <= n + p:
        for x in range(i * s - p, i * s - p + k):
            if 0 <= x < n:
                cnt[x] += 1
        i += 1
    return cnt
print("cover k4 s4 p0 (16 px):", cover(4, 4, 0, 16))
print("cover k7 s4 p3 (16 px):", cover(7, 4, 3, 16))
print("cover k3 s2 p1 (16 px):", cover(3, 2, 1, 16))
