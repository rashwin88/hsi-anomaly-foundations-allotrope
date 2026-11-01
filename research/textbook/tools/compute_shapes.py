"""Numbers for chapter 10-shapes. Run from the repo root with the project venv:
    PYTHONPATH=. python research/textbook/tools/compute_shapes.py > research/textbook/tools/compute_shapes_output.txt
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from app.foundation_models.components.spatial_encoder import SpatialEncoder
from app.foundation_models.components.seg_former_encoder import SegFormerEncoder

# 1. The 2x2x2 layout example.
x = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 2, 2)
print("cube (B=1,C=2,H=2,W=2):", x.tolist())
print("flatten(2):", x.flatten(2).tolist())
print("flatten(2).transpose(1,2):", x.flatten(2).transpose(1, 2).tolist())
print("memory after .contiguous():", x.flatten(2).transpose(1, 2).contiguous().flatten().tolist())
print("reshape(1,4,2) alone:", x.reshape(1, 4, 2).tolist())
t = x.flatten(2).transpose(1, 2)
print("round trip ok:", torch.equal(t.transpose(1, 2).reshape(1, 2, 2, 2), x))

# 2. The 3x3 input, 2x2 kernel worked convolution.
inp = torch.tensor([[1., 2., 0.], [0., 1., 3.], [2., 1., 1.]]).view(1, 1, 3, 3)
k = torch.tensor([[1., 0.], [-1., 2.]]).view(1, 1, 2, 2)
print("conv 3x3 * 2x2:", F.conv2d(inp, k).view(2, 2).tolist())

# 3. Stride 2, padding 1, kernel 3 on a 4x4 input of ones (counts how many real pixels each window sees).
ones = torch.ones(1, 1, 4, 4)
print("k3 s2 p1 on 4x4 ones, kernel of ones:", F.conv2d(ones, torch.ones(1, 1, 3, 3), stride=2, padding=1).view(2, 2).tolist())

# 4. Output sizes.
def out(h, k, s, p):
    return (h + 2 * p - k) // s + 1
for name, h, k, s, p in [("stage1", 128, 4, 4, 0), ("stage2", 32, 3, 2, 1), ("stage3", 16, 3, 2, 1),
                         ("stage4", 8, 3, 2, 1), ("stage1 if p=k//2", 128, 4, 4, 2),
                         ("stale k7 s4 p3", 128, 7, 4, 3), ("odd 33 k3 s2 p1", 33, 3, 2, 1),
                         ("SpatialEncoder 128", 128, 4, 2, 1)]:
    real = nn.Conv2d(1, 1, k, s, p)(torch.zeros(1, 1, h, h)).shape[-1]
    print(f"{name}: formula {out(h, k, s, p)}  torch {real}")
h = 100
for _ in range(3):
    h = out(h, 4, 2, 1)
    print("SpatialEncoder on 100, after stage:", h)

# 5. Receptive field of the patch-embedding chain (all-ones weights, gradient of one output).
convs = [nn.Conv2d(1, 1, 4, 4, 0, bias=False)] + [nn.Conv2d(1, 1, 3, 2, 1, bias=False) for _ in range(3)]
for c in convs:
    nn.init.constant_(c.weight, 1.0)
z = torch.zeros(1, 1, 256, 256, requires_grad=True)
y = z
for i, c in enumerate(convs):
    y = c(y)
    g, = torch.autograd.grad(y[0, 0, y.shape[-2] // 2, y.shape[-1] // 2], z, retain_graph=True)
    rows = (g[0, 0].abs().sum(1) > 0).nonzero()
    print(f"after stage {i + 1}: receptive field {int(rows.max() - rows.min() + 1)} pixels")

# 6. Parameter counts.
print("SpatialEncoder(1,32,3):", sum(p.numel() for p in SpatialEncoder(1, 32, 3).parameters()))
print("Conv2d(3,2,2):", sum(p.numel() for p in nn.Conv2d(3, 2, 2).parameters()))
print("Linear(3,2):", sum(p.numel() for p in nn.Linear(3, 2).parameters()))
print("full-image linear layer weights 165*128*128 squared:", (165 * 128 * 128) ** 2)
enc = SegFormerEncoder(32, [32, 64, 160, 256], [1, 2, 5, 8], [8, 4, 2, 1], [2, 2, 2, 2])
print("encoder feature shapes:", [tuple(f.shape) for f in enc(torch.zeros(1, 32, 128, 128))])

# 7. 1x1 conv equals a linear layer per pixel.
conv = nn.Conv2d(3, 2, 1)
lin = nn.Linear(3, 2)
lin.weight.data = conv.weight.data.view(2, 3).clone()
lin.bias.data = conv.bias.data.clone()
a = torch.randn(1, 3, 5, 5)
print("1x1 conv == linear on tokens:",
      torch.allclose(conv(a).flatten(2).transpose(1, 2), lin(a.flatten(2).transpose(1, 2)), atol=1e-6))

# 8. Exercise checks.
inp2 = torch.tensor([[2., 0., 1.], [1., 3., 0.], [0., 1., 2.]]).view(1, 1, 3, 3)
k2 = torch.tensor([[0., 1.], [1., -1.]]).view(1, 1, 2, 2)
print("exercise conv:", F.conv2d(inp2, k2).view(2, 2).tolist())
print("k7 s4 p3 on 128:", out(128, 7, 4, 3), " k3 s2 p1 on 31:", out(31, 3, 2, 1))
from app.foundation_models.components.spatial_auto_encoder import SpatialAutoencoder
import inspect
print("SpatialAutoencoder signature:", inspect.signature(SpatialAutoencoder.__init__))
print("SpatialAutoencoder(1,32,3) parameters:", sum(p.numel() for p in SpatialAutoencoder(1, 32, 3).parameters()))
