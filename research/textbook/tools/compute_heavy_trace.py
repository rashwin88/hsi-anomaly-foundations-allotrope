"""Shape trace and parameter count of Indradhanu Heavy as served (v2.1.0 = FTH3 epoch 5), for chapter 41.

Run from the repo root:
    PYTHONPATH=. PYTHONUTF8=1 C:/av/Scripts/python.exe research/textbook/tools/compute_heavy_trace.py

Reads the shipped manifest and checkpoint from C:/es/ship/fth3/spectral_stem_finetune/ (the files copied to the stack's
/models/spectral_stem_finetune/ on 2026-10-29) and PRISMA bench scene set1 from C:/Users/FITT/Documents/stage1_datasets,
loaded exactly as the evaluator does (app.evaluation.datasets.load_prisma_scene, mode "native").
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "backend")
from allotrope.foundation_models.resolver import _from_manifest
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking

SHIP = Path("C:/es/ship/fth3/spectral_stem_finetune")
BENCH = "C:/Users/FITT/Documents/stage1_datasets"
torch.manual_seed(0)


def n(m):
    return sum(p.numel() for p in m.parameters())


rm = _from_manifest(SHIP, SHIP / "current.json")
cfg = rm.model_config
man = json.loads((SHIP / "current.json").read_text(encoding="utf-8"))
print("== manifest", man["current"]["file"], "version", man["current"]["version"], "params (manifest)",
      man["current"]["params"])
keys = ("stem", "compressed_channels", "token_dim", "attention_heads", "bin_nm", "band_weighting", "embed_dims",
        "num_heads", "reduction_ratios", "num_blocks", "decoder_dim", "decoder_type", "decoder_stage_dims",
        "expansion_ratio", "seg_classes", "seg_decoder_dim", "mask_ratio", "mask_block", "drop_rate")
print("  config", {k: cfg[k] for k in keys})
print("  resolver: patch", rm.default_patch_size, "stride", rm.default_stride, "batch", rm.default_batch_size,
      "method", rm.default_scoring_method, "band grid", rm.band_grid, "exclude", rm.exclude_classes,
      "thin cloud is cloud", rm.thin_cloud_is_cloud)
sd = torch.load(SHIP / man["current"]["file"], map_location="cpu", weights_only=False)["model_state_dict"]
model = SpectralStemSegFormerMAE(
    stem=cfg["stem"], stats_wavelengths_nm=sd["stats.wl"].tolist(), stats_mean=sd["stats.mean"].tolist(),
    stats_std=sd["stats.std"].tolist(), bands=cfg["bands"], compressed_channels=cfg["compressed_channels"],
    token_dim=cfg["token_dim"], attention_heads=cfg["attention_heads"], gate_hidden=cfg["gate_hidden"],
    bin_nm=cfg["bin_nm"], band_weighting=cfg["band_weighting"], seg_classes=cfg["seg_classes"],
    seg_decoder_dim=cfg["seg_decoder_dim"], embed_dims=cfg["embed_dims"], num_heads=cfg["num_heads"],
    reduction_ratios=cfg["reduction_ratios"], num_blocks=cfg["num_blocks"], decoder_dim=cfg["decoder_dim"],
    expansion_ratio=cfg["expansion_ratio"], drop_rate=0.0, scale_decoder_dim=cfg["scale_decoder_dim"],
    decoder_type=cfg["decoder_type"], decoder_stage_dims=cfg["decoder_stage_dims"])
print("  strict load:", model.load_state_dict(sd, strict=True))
model.eval()

print("== parameters")
tot = n(model)
print(f"  total trainable {tot:,}; without seg head {tot - n(model.seg_head):,}")
for name, child in model.named_children():
    print(f"  {name}: {n(child):,}")
print("  spectral_in parts:", {k: n(v) for k, v in model.spectral_in.named_children()},
      "queries", model.spectral_in.queries.numel())
print("  spectral_out parts:", {k: n(v) for k, v in model.spectral_out.named_children()})
print("  seg head parts:", {k: n(v) for k, v in model.seg_head.named_children()},
      "projections", [n(p) for p in model.seg_head.linear_projections], "refine", [n(m) for m in model.seg_head.refine])
print("  decoder parts:", {k: n(v) for k, v in model.decoder.named_children()})
bufs = {k: v.numel() for k, v in model.state_dict().items() if k not in dict(model.named_parameters())}
print("  buffers:", bufs)
print("  state_dict numel minus stats.* and trained_wl:",
      sum(v.numel() for k, v in sd.items() if not k.startswith(("stats.", "trained_wl"))))
print("  stats curve length", sd["stats.wl"].numel(), "range", round(sd["stats.wl"].min().item(), 1),
      round(sd["stats.wl"].max().item(), 1), "| trained_wl length", sd["trained_wl"].numel())

print("== PRISMA set1, native bands, then trained_band_mask (tol 15 nm)")
from app.evaluation.datasets import load_prisma_scene
cube, validity, wl, spatial, gt = load_prisma_scene(BENCH, "set1", "native")
print("  cube", cube.shape, "wavelength range", round(float(wl.min()), 1), round(float(wl.max()), 1))
keep = model.trained_band_mask(wl).numpy()
print("  bands", len(wl), "kept", int(keep.sum()), "dropped", int((~keep).sum()), ":",
      [round(float(w), 1) for w in wl[~keep]])
curve = sd["stats.wl"].numpy()
kc = np.abs(wl[:, None] - curve[None, :]).min(axis=1) <= 15.0
print("  against the 202-wavelength pretraining curve alone (a pretrained model's record): kept", int(kc.sum()),
      "dropped", int((~kc).sum()), [round(float(w), 1) for w in wl[~kc]])
tw = sd["trained_wl"].numpy()
gaps = [(round(float(a), 1), round(float(b), 1)) for a, b in zip(tw[:-1], tw[1:]) if b - a > 30]
print("  gaps over 30 nm in trained_wl:", gaps, "| trained_wl max", round(float(tw.max()), 1))
wlk = torch.tensor(wl[keep], dtype=torch.float32)
N = int(keep.sum())
G = model.spectral_in.n_bins
onehot_bins = torch.unique(model.spectral_in.bin_index(wlk))
print("  bins", G, "occupied by set1's kept bands", len(onehot_bins),
      "max bands in one bin", int(torch.bincount(model.spectral_in.bin_index(wlk)).max()))

print("== shape trace, one masked training-style pass: B=2, 128 x 128, N =", N, ", mask ratio", cfg["mask_ratio"])
B, H = 2, 128
x = 0.1 + 0.02 * torch.rand(B, N, H, H)
bv = torch.ones(B, N, H, H, dtype=torch.bool)
tv = TokenMasking.pixel_mask_to_token_mask(torch.ones(B, 1, H, H), kernel_size=4, stride=4, padding=0)
keep_mask, pred = TokenMasking.generate_prediction_mask(tv, cfg["mask_ratio"], "cpu")
print("  tokens", tv.shape[1], "hidden", int(pred[0].sum()), "kept", int(keep_mask[0].sum()))
log = []


def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        log.append((name, tuple(inp[0].shape) if inp else None, tuple(o.shape)))
    return f


si = model.spectral_in
for nm in ("band_weight", "skip_weight", "bin_bias", "k", "v", "q", "o", "to_channel"):
    getattr(si, nm).register_forward_hook(hook(f"stem.{nm}"))
si.register_forward_hook(hook("stem"))
model.norm.register_forward_hook(hook("norm(BN)"))
enc = model.encoder
for i in range(4):
    enc.patch_embeds[i].register_forward_hook(hook(f"enc{i+1}.patch_embed"))
    enc.blocks[i][0].attn.register_forward_hook(hook(f"enc{i+1}.attn"))
    enc.blocks[i][0].ffn.fc1.register_forward_hook(hook(f"enc{i+1}.fc1"))
    enc.norms[i].register_forward_hook(hook(f"enc{i+1}.norm"))
dec = model.decoder
for i, p in enumerate(dec.linear_projections):
    p.register_forward_hook(hook(f"dec.proj{i+1}"))
dec.fuse_conv.register_forward_hook(hook("dec.fuse"))
dec.up_half.register_forward_hook(hook("dec.up_half"))
dec.up_full.register_forward_hook(hook("dec.up_full"))
dec.head.register_forward_hook(hook("dec.head"))
model.spectral_out.out_weight.register_forward_hook(hook("out.out_weight"))
model.spectral_out.register_forward_hook(hook("spectral_out"))
with torch.no_grad():
    rec = model(x, wlk, bv, keep_mask=keep_mask)
print("  reconstruction", tuple(rec.shape))
for e in log:
    print("   ", e)
log.clear()
sh = model.seg_head
for i, p in enumerate(sh.linear_projections):
    p.register_forward_hook(hook(f"seg.proj{i+1}"))
sh.fuse_conv.register_forward_hook(hook("seg.fuse"))
sh.refine[0].register_forward_hook(hook("seg.refine0"))
sh.refine[2].register_forward_hook(hook("seg.refine2"))
with torch.no_grad():
    logits = model.segment(x, wlk, bv)
print("  segment logits", tuple(logits.shape))
for e in log:
    if e[0].startswith("seg"):
        print("   ", e)

print("== sizes worth quoting")
P = B * H * H
print(f"  stem rows P = {B}*{H}*{H} = {P:,}; attention weights if stored: P*4*64*53 = {P*4*64*53:,}"
      f" ({P*4*64*53*4/2**20:.0f} MiB float32)")
k1 = int(keep_mask[0].sum())
for s, (C, R, h) in enumerate(zip(cfg["embed_dims"], cfg["reduction_ratios"], cfg["num_heads"])):
    side = H // (4 * 2 ** s)
    Ntok = side * side
    keys = (side // R) ** 2 if R > 1 else Ntok
    print(f"  stage {s+1}: grid {side}x{side} = {Ntok} tokens, C {C}, heads {h}, head dim {C//h}, keys {keys}")
print(f"  masked stage 1: full attention among {k1} kept tokens, {k1*k1:,} scores per head, x3 heads")
print("  F1 values per item:", 96 * 32 * 32, "| input values per item:", N * H * H, "| latent z per item:", 64 * H * H)
