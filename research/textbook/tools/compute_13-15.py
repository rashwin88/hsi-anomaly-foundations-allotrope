"""Book chapters ch:attention, ch:esa, ch:block - every number the three chapters compute.
Builds the real blocks on CPU (no training) and checks the hand calculations.
Run: C:/av/Scripts/python.exe research/textbook/tools/compute_13-15.py > research/textbook/tools/compute_13-15_output.txt"""
import itertools, math, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../../..")))
from app.foundation_models.components.segformer_block import SegFormerBlock
from app.foundation_models.components.efficient_self_attention import EfficientSelfAttention

def n(m): return sum(p.numel() for p in m.parameters())
def sm(s):
    m = max(s); z = [math.exp(x - m) for x in s]; t = sum(z); return [x / t for x in z]

print("== ch:attention ==")
s = [2, 1, 0]
print("softmax [2,1,0]: exps", [round(math.exp(x), 4) for x in s], "sum", round(sum(math.exp(x) for x in s), 4),
      "weights", [round(w, 4) for w in sm(s)])
print("shifted [0,-1,-2]: exps", [round(math.exp(x), 4) for x in [0, -1, -2]],
      "sum", round(sum(math.exp(x) for x in [0, -1, -2]), 4))
print("exp(20)", math.exp(20), "exp(-15)", math.exp(-15), "softmax[20,0,-15]", sm([20, 0, -15]))
print("softmax[1000,999]", sm([1000, 999]), "float32 max", torch.finfo(torch.float32).max,
      "log(max)", math.log(torch.finfo(torch.float32).max))
print("torch exp(89.) float32", torch.exp(torch.tensor(89.0)))
p = sm([20, 0, -15]); print("jacobian diag p(1-p)", [x * (1 - x) for x in p])
p = sm([1, 0, -1]); print("softmax[1,0,-1]", p, "p(1-p)", [x * (1 - x) for x in p])

X = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
for name, WQ, WK, WV in [("identity", torch.eye(2), torch.eye(2), torch.eye(2)),
                         ("projected", torch.tensor([[1., 0.], [1., 1.]]), torch.eye(2),
                          torch.tensor([[0., 1.], [1., 0.]]))]:
    Q, K, V = X @ WQ, X @ WK, X @ WV
    S = Q @ K.T; Ss = S / math.sqrt(2); A = torch.softmax(Ss, -1); O = A @ V
    print(name, "Q", Q.tolist(), "K", K.tolist(), "V", V.tolist())
    print("  QK^T", S.tolist()); print("  scaled", Ss.numpy().round(4).tolist())
    print("  exp", torch.exp(Ss).numpy().round(4).tolist(), "rowsums", torch.exp(Ss).sum(-1).numpy().round(4).tolist())
    print("  A", A.numpy().round(4).tolist()); print("  out", O.numpy().round(4).tolist())
# permutation: swap tokens 1 and 2 in identity case
P = torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
Xp = P @ X; Ap = torch.softmax(Xp @ Xp.T / math.sqrt(2), -1); print("permuted out", (Ap @ Xp).numpy().round(4).tolist())

vals = []
for q in itertools.product([-1, 1], repeat=2):
    for k in itertools.product([-1, 1], repeat=2):
        vals.append(sum(a * b for a, b in zip(q, k)))
from collections import Counter
print("d=2 +-1 dot products", sorted(Counter(vals).items()), "var", sum(v * v for v in vals) / len(vals))
g = torch.Generator().manual_seed(0)
for d in [2, 16, 32, 64]:
    q = torch.randn(200000, d, generator=g); k = torch.randn(200000, d, generator=g)
    dot = (q * k).sum(-1)
    print(f"d={d} sample var {dot.var().item():.3f} std {dot.std().item():.3f} sqrt(d) {math.sqrt(d):.3f}"
          f" scaled var {(dot / math.sqrt(d)).var().item():.3f}")
print("exp(11.31)", math.exp(2 * math.sqrt(32)), "exp(2)", math.exp(2))
for C in [32, 64, 160, 256]:
    print("4C^2+4C C=%d" % C, 4 * C * C + 4 * C)
for h in [1, 2]:
    print("ESA C=32 R=8 heads", h, n(EfficientSelfAttention(32, h, 8)))

print("== ch:esa ==")
B = 128  # configs/hyperspectral_segformer_exp_2.json batch_size
for (C, h, R, H) in [(32, 2, 8, 32), (64, 2, 4, 16), (160, 5, 2, 8), (256, 8, 1, 4)]:
    N = H * H; Hr = H // R; Nr = Hr * Hr; d = C // h
    plain_scores, red_scores = N * N, N * Nr
    mac_plain = 4 * N * C * C + 2 * N * N * C
    mac_red = N * C * C + (N * C * C if R > 1 else 0) + 2 * Nr * C * C + N * C * C + 2 * N * Nr * C
    mem_plain = B * h * N * N * 4; mem_red = B * h * N * Nr * 4
    print(f"stage C={C} h={h} d={d} R={R} grid {H}x{H} N={N} reduced {Hr}x{Hr} Nr={Nr} scores {red_scores} vs {plain_scores}"
          f" ratio {plain_scores // red_scores}")
    print(f"   MACs plain {mac_plain:,} reduced {mac_red:,}  QK+AV plain {2*N*N*C:,} reduced {2*N*Nr*C:,}"
          f"  conv {N*C*C if R>1 else 0:,}")
    print(f"   score memory float32 per image per head plain {N*N*4:,} B reduced {N*Nr*4:,} B;"
          f" batch {B} x heads {h}: plain {mem_plain:,} B ({mem_plain/2**20:.1f} MiB) reduced {mem_red:,} B ({mem_red/2**20:.2f} MiB)")
    e = EfficientSelfAttention(C, h, R)
    parts = {k: n(v) for k, v in e.named_children()}
    print("   ESA params", n(e), parts)
    x = torch.randn(2, N, C); print("   ESA out shape", tuple(e(x, H, H).shape))
print("masked survivors 1024*0.35", 1024 * 0.35, "358^2", 358 ** 2, "512^2", 512 ** 2)
e = EfficientSelfAttention(32, 2, 8); print("masked ESA out", tuple(e(torch.randn(1, 358, 32), 32, 32).shape))
G = torch.tensor([[1., 2., 0., 1.], [3., 2., 1., 0.], [0., 0., 4., 4.], [0., 0., 4., 4.]])
avg = F.conv2d(G[None, None], torch.full((1, 1, 2, 2), 0.25), stride=2); print("toy avg summary", avg[0, 0].tolist())
pick = F.conv2d(G[None, None], torch.tensor([[[[1., 0.], [0., 0.]]]]), stride=2); print("toy pick summary", pick[0, 0].tolist())
k = avg.flatten().tolist(); print("toy query 1 scores", k, "softmax", [round(w, 4) for w in sm(k)],
                                  "out", round(sum(w * v for w, v in zip(sm(k), k)), 4))
print("conv output size floor((H-R)/R)+1 for H=32,R=8:", (32 - 8) // 8 + 1, " H=30,R=8:", (30 - 8) // 8 + 1)

print("== ch:block ==")
x = torch.tensor([[1., 2., 3., 4.]]); ln = nn.LayerNorm(4); print("LN [1,2,3,4]", ln(x).detach().numpy().round(4).tolist(), "eps", ln.eps)
print("1/sqrt(1.25)", 1 / math.sqrt(1.25), "sqrt(1.25)", math.sqrt(1.25), "sqrt(1.25+1e-5)", math.sqrt(1.25 + 1e-5))
ln.weight.data = torch.tensor([2., 1., 1., 0.5]); ln.bias.data = torch.tensor([0., 0., 1., 0.])
print("LN affine", ln(x).detach().numpy().round(4).tolist())
ln2 = nn.LayerNorm(4); print("LN 3x+10", ln2(3 * x + 10).detach().numpy().round(4).tolist())
print("GELU", nn.GELU()(torch.tensor([-3., -1., 0., 1., 3.])).numpy().round(4).tolist(),
      "tanh approx", nn.GELU(approximate="tanh")(torch.tensor([-1., 0., 1.])).numpy().round(4).tolist())
print("Phi(1)", 0.5 * (1 + math.erf(1 / math.sqrt(2))), "phi(1)", math.exp(-0.5) / math.sqrt(2 * math.pi))
for xv in [-1., 0., 1.]:
    t = torch.tensor(xv, requires_grad=True); nn.GELU()(t).backward(); print("GELU'", xv, round(t.grad.item(), 4))
print("residual products 0.5^20", 0.5 ** 20, "0.9^20", 0.9 ** 20, "1.05^20", 1.05 ** 20, "0.95^20", 0.95 ** 20)
G3 = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
lap = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]])
print("dwconv toy laplacian pad1", F.conv2d(G3[None, None], lap, padding=1)[0, 0].tolist())
for C in [32, 64, 160, 256]:
    h4 = 4 * C
    print(f"Mix-FFN C={C}: fc1 {C*h4+h4:,} dw {h4*9+h4:,} fc2 {h4*C+C:,} ordinary3x3 {h4*h4*9+h4:,}")
for (C, h, R) in [(32, 2, 8), (64, 2, 4), (160, 5, 2), (256, 8, 1), (96, 3, 8)]:
    b = SegFormerBlock(C, h, R)
    print(f"block C={C} R={R}: total {n(b):,}", {k: n(v) for k, v in b.named_children()},
          "x2 =", f"{2*n(b):,}")
    for k, v in b.named_modules():
        if k and len(list(v.children())) == 0 and n(v): print("     ", k, n(v))
print("stage blocks sum x2", 2 * sum(n(SegFormerBlock(C, h, R)) for C, h, R in [(32, 2, 8), (64, 2, 4), (160, 5, 2), (256, 8, 1)]))
drop = nn.Dropout(0.4); print("dropout scale 1/(1-0.4)", 1 / 0.6, "eval identity", drop.eval()(torch.ones(3)).tolist())
xd = torch.tensor([0.5, -1.0, 2.0, 0.3, 1.2]); md = torch.tensor([1., 0., 1., 1., 0.])
print("dropout toy", (xd * md / 0.6).numpy().round(4).tolist())
