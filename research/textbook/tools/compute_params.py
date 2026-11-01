"""Book chapter ch:params - build the real models on CPU and count parameters (no training).
Run: C:/av/Scripts/python.exe research/textbook/tools/compute_params.py > research/textbook/tools/compute_params_output.txt"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, __import__("os").path.abspath(__import__("os").path.join(__file__, "../../../..")))
from app.foundation_models.components.hyperspectral_seg_former_mae import HyperspectralSegFormerMAE
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
def n(m): return sum(p.numel() for p in m.parameters())
m = HyperspectralSegFormerMAE(165, 32, num_heads=[2,2,5,8], pixel_mean=[0.1]*165, pixel_std=[0.08]*165)
print("total", n(m), "buffers", sum(b.numel() for b in m.buffers()))
for name, ch in m.named_children(): print(" ", name, n(ch))
e = m.encoder
print(" pe", [n(p) for p in e.patch_embeds], "blocks", [n(b) for b in e.blocks], "norms", [n(x) for x in e.norms])
blk = e.blocks[0][0]
for name, ch in blk.named_children(): print("   blk0", name, n(ch))
for name, ch in blk.named_modules(): 
    if name and "." in name and len(list(ch.children()))==0: print("    ", name, n(ch))
d = m.decoder
print(" dec proj", [n(p) for p in d.linear_projections], "fuse", n(d.fuse_conv), "refine", [n(x) for x in d.refine])
m24 = HyperspectralSegFormerMAE(165, 24, num_heads=[1,2,5,8], pixel_mean=[0.1]*165, pixel_std=[0.08]*165)
print("D24", n(m24), n(m24)+sum(b.numel() for b in m24.buffers()))
wl = [400+10*i for i in range(202)]
s = SpectralStemSegFormerMAE("reduced_attention", wl, [0.1]*202, [0.08]*202, bands=202, num_heads=(2,2,5,8), seg_classes=5, band_weighting="width")
print("stem total", n(s))
for name, ch in s.named_children(): print(" ", name, n(ch))
s.eval()
x = torch.rand(2,202,128,128)
with torch.no_grad():
    keep = torch.ones(2,1024); keep[:, ::3] = 0
    out = s(x, torch.tensor(wl, dtype=torch.float32), keep_mask=keep)
print(out.shape)
# shapes trace for m
m.eval()
feats=[]
with torch.no_grad():
    x = torch.rand(4,165,128,128); mask=torch.ones(4,1,128,128)
    from app.foundation_models.components.token_masking import TokenMasking as TM
    tm = TM.pixel_mask_to_token_mask(mask,4,4,0)
    keep,pred = TM.generate_prediction_mask(tm,0.65,'cpu')
    print("kept per item", keep.sum(1))
    f = m.encoder(m.compressor(m.normalize(x*mask)), keep_mask=keep)
    print([tuple(t.shape) for t in f])
    print(tuple(m(x,mask,keep).shape))

# --- which parameters get no gradient in a masked pass ---
torch.manual_seed(0)
m = HyperspectralSegFormerMAE(165, 32, num_heads=[2,2,5,8], pixel_mean=[0.1]*165, pixel_std=[0.08]*165)
x = torch.rand(2,165,128,128); mask=torch.ones(2,1,128,128)
keep,pred = TM.generate_prediction_mask(TM.pixel_mask_to_token_mask(mask,4,4,0),0.65,'cpu')
m(x,mask,keep).abs().mean().backward()
idle=0
for n,p in m.named_parameters():
    if p.grad is None or p.grad.abs().sum()==0:
        print("no grad:", n, p.numel()); idle+=p.numel()
print("idle", idle)

# --- Indradhanu Heavy (FTH3 config) for the drift note: build only, CPU, no weights ---
h = SpectralStemSegFormerMAE("reduced_attention", wl, [0.1]*202, [0.08]*202, bands=202, compressed_channels=64,
                             token_dim=32, attention_heads=4, band_weighting="width", seg_classes=5, seg_decoder_dim=256,
                             embed_dims=(96, 192, 480, 768), num_heads=(3, 6, 10, 12), decoder_dim=512,
                             decoder_type="progressive", decoder_stage_dims=(256, 128))
count = lambda mod: sum(p.numel() for p in mod.parameters())
print("heavy total", count(h))
for name, ch in h.named_children(): print(" ", name, count(ch))

# --- multiply-adds of Conv2d and Linear layers for one unmasked 128x128 patch (attention matmuls excluded) ---
import torch.nn as nn
macs = {}
def hook(name):
    def f(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            k = mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
            macs[name] = out.numel() * k
        else:
            macs[name] = out.numel() // out.shape[-1] * mod.in_features * mod.out_features
    return f
for name, mod in m.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear)): mod.register_forward_hook(hook(name))
m.eval()
with torch.no_grad(): m(torch.rand(1, 165, 128, 128))
groups = {}
for k, v in macs.items():
    g = k.split(".")[0] if not k.startswith("encoder") else ".".join(k.split(".")[:3])
    groups[g] = groups.get(g, 0) + v
for g, v in groups.items(): print("macs", g, v)
print("macs total conv+linear", sum(macs.values()))
