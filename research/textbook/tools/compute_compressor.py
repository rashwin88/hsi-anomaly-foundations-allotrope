"""Numbers for chapter 11-compressor. Run from the repo root with the project venv:
    PYTHONPATH=. python research/textbook/tools/compute_compressor.py > research/textbook/tools/compute_compressor_output.txt
"""
import torch
import torch.nn as nn

from app.foundation_models.components.spectral_compressor import SpectralCompressor, SpectralDecompressor

comp, decomp = SpectralCompressor(165, 32), SpectralDecompressor(32, 165)
nc = sum(p.numel() for p in comp.parameters())
nd = sum(p.numel() for p in decomp.parameters())
print("compressor params:", nc, " decompressor params:", nd, " total:", nc + nd)
print("fraction of 5,506,629:", (nc + nd) / 5506629)
print("compressor buffers:", {k: tuple(v.shape) for k, v in comp.state_dict().items() if "running" in k or "num_batches" in k})
print("Conv2d(165,24,1):", sum(p.numel() for p in nn.Conv2d(165, 24, 1).parameters()))
print("stage-1 conv weights 165->32 k4:", 165 * 32 * 16, " 32->32 k4:", 32 * 32 * 16)

# 4-value BatchNorm example (one channel, 4 values in the batch).
bn = nn.BatchNorm2d(1)
print("eps, momentum:", bn.eps, bn.momentum)
x = torch.tensor([1., 2., 3., 6.]).view(4, 1, 1, 1)
bn.train()
print("train-mode output:", [round(v, 4) for v in bn(x).flatten().tolist()])
print("running_mean, running_var after one step:", bn.running_mean.item(), round(bn.running_var.item(), 5))
bn.eval()
print("eval-mode output for the same values:", [round(v, 4) for v in bn(x).flatten().tolist()])
with torch.no_grad():
    bn.weight.fill_(2.0); bn.bias.fill_(0.5)
bn2 = nn.BatchNorm2d(1)
with torch.no_grad():
    bn2.weight.fill_(2.0); bn2.bias.fill_(0.5)
print("train-mode with gamma=2, beta=0.5:", [round(v, 4) for v in bn2(x).flatten().tolist()])

# The conv bias in front of BatchNorm has no effect in train mode.
conv = nn.Conv2d(3, 2, 1)
bn3 = nn.BatchNorm2d(2)
a = torch.randn(4, 3, 5, 5)
y1 = bn3(conv(a))
with torch.no_grad():
    conv.bias.add_(10.0)
y2 = bn3(conv(a))
print("bias shift changes train-mode BN output:", not torch.allclose(y1, y2, atol=1e-5))

# The 3 -> 2 -> 3 linear bottleneck.
W = torch.tensor([[1., 0., 0.], [0., .5, .5]])
V = torch.tensor([[1., 0.], [0., 1.], [0., 1.]])
xv = torch.tensor([0.2, 0.4, 0.6])
print("y = W x:", (W @ xv).tolist(), " xhat = V W x:", (V @ W @ xv).tolist())
print("V W =", (V @ W).tolist(), " rank:", torch.linalg.matrix_rank(V @ W).item())
print("W (0,1,-1) =", (W @ torch.tensor([0., 1., -1.])).tolist())
print("rank of real compressor weight (random init):", torch.linalg.matrix_rank(comp.compress.weight.view(32, 165)).item())
