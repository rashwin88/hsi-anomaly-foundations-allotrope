"""Book chapters ch:masking and ch:decoder - the small examples, checked with the real torch ops.
Run: C:/av/Scripts/python.exe research/textbook/tools/compute_masking.py > research/textbook/tools/compute_masking_output.txt"""
import torch, torch.nn.functional as F
x = torch.arange(8.).reshape(1,8,1,1)
print("ps r2 C2:", F.pixel_shuffle(x,2)[0])
y = torch.tensor([[1.,3.],[5.,7.]]).reshape(1,1,2,2)
print("bilinear 2->4:", F.interpolate(y,size=(4,4),mode='bilinear',align_corners=False)[0,0])
z = torch.tensor([[0.,4.]]).reshape(1,1,1,2)
print("1d 2->4:", F.interpolate(z,size=(1,4),mode='bilinear',align_corners=False)[0,0])
pm = torch.tensor([[1,1,1,0],[1,1,0,0],[1,0,1,1],[1,0,1,1]],dtype=torch.float).reshape(1,1,4,4)
print("avgpool (book 4x4 example):", F.avg_pool2d(pm,2,2,0))
print("argsort desc of [0,1,0,1,0,1]:", torch.tensor([[0,1,0,1,0,1.]]).argsort(dim=1,descending=True))
print("argsort desc of [1,0,1,1,0,1] (docstring):", torch.tensor([[1,0,1,1,0,1.]]).argsort(dim=1,descending=True))
print("even kernel 4 erosion output shape on 8x8:", F.max_pool2d(torch.zeros(1,1,8,8),4,1,2).shape)
# exercise: v=[1,1,1,0,1,1], noise [0.9,0.1,0.05,0.4,0.6,0.02], rho 0.65
v = torch.tensor([[1,1,1,0,1,1.]]); nz = torch.tensor([[0.9,0.1,0.05,0.4,0.6,0.02]]) + (1-v)*2
pi = nz.argsort(dim=1); nh = (v.sum(1)*0.65).long()
s = (torch.arange(6).float() < nh).float().unsqueeze(0)
q = torch.zeros(1,6).scatter_(1, pi, s)
print("exercise: n_hide", nh, "pi", pi, "q", q, "inverse", pi.argsort(dim=1), "gather", s[0, pi.argsort(dim=1)[0]])
# four-mask figure: 16x16, cloud rows>=10 cols>=11, hidden tokens
cloud = torch.ones(1,1,16,16); cloud[0,0,10:,11:] = 0
tm = (F.avg_pool2d(cloud,4,4,0) > 0.5).float().flatten()
print("fig token validity:", tm.tolist())
hid = torch.tensor([1,0,1,1,0,1,0,1,1,0,1,0,0,1,1,0.]).reshape(1,1,4,4)
Q = F.interpolate(hid, size=(16,16), mode="nearest")
print("graded no erosion:", (Q*cloud).sum().item(), " graded k3:", (Q*(1-F.max_pool2d(1-cloud,3,1,1))).sum().item())
print("int(6*.65)", int(6*0.65), (torch.tensor([6.])*0.65).long())
print("int(1024*.65)", (torch.tensor([1024.])*0.65).long())
k = torch.tensor([[0,1,1,1,0,1.]])
print("argsort desc", k.argsort(dim=1,descending=True))
k2 = torch.tensor([[1,0,1,1.]])
print("argsort desc2", k2.argsort(dim=1,descending=True))
noise=torch.tensor([[0.42,0.91,0.07,0.66,0.18,0.55]])
print("argsort noise", noise.argsort(dim=1), noise.argsort(dim=1).argsort(dim=1))
m = torch.ones(1,1,5,5); m[0,0,0,:]=0; m[0,0,2,2]=0
inv = 1-m
print("eroded k3:", 1-F.max_pool2d(inv,3,1,1))
m2=torch.ones(1,1,4,4)
print("all-valid k3:", 1-F.max_pool2d(1-m2,3,1,1))

# --- ch:encoder: chance that a stage-2 window (4, 6 or 9 stage-1 tokens) holds no visible token ---
from math import comb
N, hid = 1024, int(1024 * 0.65)
for k in (4, 6, 9):
    indep = 0.65 ** k
    exact = comb(N - k, hid - k) / comb(N, hid)   # all k chosen among the hidden
    print(f"window {k}: indep {indep:.6f}  exact {exact:.6f}")
e_ind = 1 * 0.65**4 + 30 * 0.65**6 + 225 * 0.65**9
e_ex = sum(c * comb(N - k, hid - k) / comb(N, hid) for c, k in ((1, 4), (30, 6), (225, 9)))
print(f"expected empty stage-2 tokens: indep {e_ind:.4f} exact {e_ex:.4f}")
# which stage-1 rows a stage-2 token at row h reads (k=3, s=2, p=1, 32 rows)
print("rows for h=0:", [r for r in (-1, 0, 1) if 0 <= r < 32], " h=15:", [r for r in (29, 30, 31) if 0 <= r < 32])
