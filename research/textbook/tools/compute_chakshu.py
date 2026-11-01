"""Numbers for the Chakshu chapter (19a-chakshu.tex), on CPU, no training.

Run from the repo root:
    PYTHONPATH=. C:/av/Scripts/python.exe research/textbook/tools/compute_chakshu.py [MODELS_DIR]

MODELS_DIR holds a copy of the stack's /models volume. It was copied on 2026-10-31 with
    docker cp allotrope_worker:/models/segformer_mae <dir>/
(inside the WSL Ubuntu distro), to a short path because the files sit deeper than Windows' 260-character limit
under the session scratchpad. Default: C:/Users/FITT/AppData/Local/Temp/claude/m5.
"""
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
sys.path.insert(0, "backend")
from app.foundation_models.components.seg_former_mae import SegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
from app.utils.patch_generation.generate_patch_plan import PatchPlanGenerator
from app.models.patches.patching_request import PatchRequest

MODELS = Path(sys.argv[1] if len(sys.argv) > 1 else "C:/Users/FITT/AppData/Local/Temp/claude/m5")
D = MODELS / "segformer_mae"
torch.manual_seed(0)


def n(m):
    return sum(p.numel() for p in m.parameters())


print("== manifest", D / "current.json")
man = json.loads((D / "current.json").read_text(encoding="utf-8"))
print("  codename", man["codename"]["name"], "| current", man["current"])
print("  normalization", {k: man["normalization"][k] for k in ("mean", "std", "source")})

print("== resolver: the model_config the worker builds from the manifest")
from allotrope.foundation_models.resolver import _from_manifest
rm = _from_manifest(D, D / "current.json")
print("  model_config", rm.model_config)
print("  default method", rm.default_scoring_method, "| methods", rm.scoring_methods,
      "| patch", rm.default_patch_size, "| stride", rm.default_stride, "| batch", rm.default_batch_size,
      "| valid patch sizes", rm.valid_patch_sizes, "| pixel stats", rm.pixel_stats_relpath if hasattr(rm, "pixel_stats_relpath") else rm.pixel_stats_abs_path)

ck = torch.load(D / "segformer_mae_v0.8.0_epoch495.pt", map_location="cpu", weights_only=False)
cfg = ck["config"]
mc = cfg["model_config"]
print("== checkpoint v0.8.0")
print("  epoch", ck["epoch"], "train_loss", ck["train_loss"], "val_losses", ck["val_losses"], "avg_val", ck["avg_val_loss"])
print("  model_config", mc)
print("  data", {k: cfg["data"][k] for k in ("train_samples_per_epoch", "test_samples_per_epoch", "num_epochs",
                                              "data_provider", "batch_size", "pixel_stats_path")})
print("  lr", cfg["learning_rate"], cfg["lr_schedule"])
sd = ck["model_state_dict"]
print("  buffers", {k: sd[k].flatten().tolist() for k in sd if "normalize" in k})
print("  state_dict numel", sum(v.numel() for v in sd.values()))
same = all(rm.model_config[k] == mc[k] for k in ("embed_dims", "num_heads", "reduction_ratios", "num_blocks",
                                                 "decoder_dim", "expansion_ratio", "in_channels"))
print("  resolver config == checkpoint config (architecture keys):", same)

mean, std = sd["normalize.mean"].item(), sd["normalize.std"].item()
model = SegFormerMAE(in_channels=1, embed_dims=mc["embed_dims"], num_heads=mc["num_heads"],
                     reduction_ratios=mc["reduction_ratios"], num_blocks=mc["num_blocks"],
                     decoder_dim=mc["decoder_dim"], expansion_ratio=mc["expansion_ratio"], drop_rate=0.0,
                     pixel_mean=[mean], pixel_std=[std])
res = model.load_state_dict(sd, strict=True)
print("  strict load:", res)
model.eval()

print("== parameters (trainable, from .parameters())")
print("  total", f"{n(model):,}", " encoder", f"{n(model.encoder):,}", " decoder", f"{n(model.decoder):,}")
enc = model.encoder
for i in range(4):
    print(f"  stage {i+1}: patch_embed {n(enc.patch_embeds[i]):,}  block {n(enc.blocks[i][0]):,}  "
          f"blocks {n(enc.blocks[i]):,}  norm {n(enc.norms[i]):,}")
    b = enc.blocks[i][0]
    print("     block parts:", {k: n(v) for k, v in b.named_children()},
          " attn parts:", {k: n(v) for k, v in b.attn.named_children()},
          " ffn parts:", {k: n(v) for k, v in b.ffn.named_children()})
print("  decoder parts:", {k: n(v) for k, v in model.decoder.named_children()},
      " projections:", [n(p) for p in model.decoder.linear_projections],
      " refine:", [n(m) for m in model.decoder.refine])
print("  buffers:", sum(b.numel() for b in model.buffers()))


def block_formula(C, R, E):
    c2 = 4 + (R * R if R > 1 else 0) + 2 * E
    c1 = 2 + 4 + (1 + 2 if R > 1 else 0) + 2 + E + 9 * E + E + 1
    return c2, c1, c2 * C * C + c1 * C


for C, R in zip(mc["embed_dims"], mc["reduction_ratios"]):
    print(f"  block formula C={C} R={R} E=2:", block_formula(C, R, 2))

print("== the v0.1.0 alternative against today's code")
ck1 = torch.load(D / "segformer_mae_v0.1.0_epoch10.pt", map_location="cpu", weights_only=False)
mc1 = ck1["config"]["model_config"]
m1 = SegFormerMAE(in_channels=1, embed_dims=mc1["embed_dims"], num_heads=mc1["num_heads"],
                  reduction_ratios=mc1["reduction_ratios"], num_blocks=mc1["num_blocks"],
                  decoder_dim=mc1["decoder_dim"], expansion_ratio=mc1["expansion_ratio"], pixel_mean=[mean],
                  pixel_std=[std])
try:
    m1.load_state_dict(ck1["model_state_dict"], strict=True)
    print("  v0.1.0 loads")
except RuntimeError as e:
    msg = str(e).replace("\n", " ")
    print("  v0.1.0 strict load FAILS:", msg[:900])
print("  v0.1.0 stage-1 kernel", tuple(ck1["model_state_dict"]["encoder.patch_embeds.0.proj.weight"].shape),
      "| today's code builds", tuple(m1.encoder.patch_embeds[0].proj.weight.shape))
print("  v0.1.0 state_dict numel", sum(v.numel() for v in ck1["model_state_dict"].values()),
      "epoch", ck1["epoch"], "val", ck1["val_losses"])

print("== shape trace: training pass, B=8, 256 x 256, mask ratio", mc["mask_ratio"])
B, H = 8, 256
x = 24.0 + 5.0 * torch.randn(B, 1, H, H)
valid = torch.ones(B, 1, H, H)
tok_valid = TokenMasking.pixel_mask_to_token_mask(valid, kernel_size=4, stride=4, padding=0)
keep, pred = TokenMasking.generate_prediction_mask(tok_valid, mc["mask_ratio"], "cpu")
print("  tokens", tok_valid.shape[1], "hidden per item", int(pred[0].sum()), "kept", int(keep[0].sum()))
shapes = []


def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        shapes.append((name, tuple(inp[0].shape), tuple(o.shape)))
    return f


for i in range(4):
    enc.patch_embeds[i].register_forward_hook(hook(f"stage{i+1}.patch_embed"))
    enc.blocks[i][0].register_forward_hook(hook(f"stage{i+1}.block"))
    enc.blocks[i][0].attn.register_forward_hook(hook(f"stage{i+1}.attn"))
    enc.blocks[i][0].ffn.fc1.register_forward_hook(hook(f"stage{i+1}.ffn.fc1"))
for i, p in enumerate(model.decoder.linear_projections):
    p.register_forward_hook(hook(f"dec.proj{i+1}"))
model.decoder.fuse_conv.register_forward_hook(hook("dec.fuse"))
model.decoder.refine[0].register_forward_hook(hook("dec.refine0"))
model.decoder.refine[2].register_forward_hook(hook("dec.refine2"))
model.register_forward_hook(hook("model"))
with torch.no_grad():
    feats = enc(model.normalize(x), keep_mask=keep)
    out = model(x, mask=valid, keep_mask=keep)
for f in feats:
    print("  feature", tuple(f.shape))
seen = set()
for name, i, o in shapes:
    if name not in seen:
        print(f"  {name:20s} in {i}  out {o}")
        seen.add(name)

print("== attention table sizes at 256 x 256 (per head, per patch)")
for s, (C, R, h) in enumerate(zip(mc["embed_dims"], mc["reduction_ratios"], mc["num_heads"])):
    side = H // (4 * 2 ** s)
    N = side * side
    keys = (side // R) ** 2 if R > 1 else N
    print(f"  stage {s+1}: grid {side}x{side} tokens {N}, keys {keys}, heads {h}, head dim {C//h}, "
          f"scores per head {N*keys:,}")
kept = int(keep[0].sum())
print(f"  masked stage 1 (training): full attention over kept {kept}: {kept*kept:,} per head")
print(f"  masked stage 1 (inference checkerboard): kept 2048: {2048*2048:,} per head")

print("== normalisation")
for v, lab in ((0.0, "invalid pixel zeroed (0 C)"), (35.0, "35 C"), (24.0, "24 C"), (5000.0, "HotSat DN 5000"),
               (5400.0, "HotSat DN 5400")):
    print(f"  {lab}: z = ({v} - {mean:.4f}) / {std:.4f} = {(v-mean)/std:.4f}")
print("  course's example (5000-290)/10 =", (5000 - 290) / 10)
print("  Kelvin of the mean:", mean + 273.15)

print("== trimmed L1 example (trim 0.05)")
errs = [0.10, 0.20, 0.15, 0.30, 0.05, 0.25, 0.20, 0.10, 0.15, 0.10, 0.20, 0.30, 0.25, 0.10, 0.15, 0.20, 0.05,
        0.10, 0.20, 9.00]
k = int(len(errs) * (1 - 0.05))
srt = sorted(errs)
print("  n", len(errs), "keep", k, "plain mean", sum(errs) / len(errs), "trimmed mean", sum(srt[:k]) / k)

print("== tiling (PatchPlanGenerator): patch 256, stride 128")
for hgt, wid in ((1000, 1000), (7801, 7681)):
    plan = PatchPlanGenerator().generate_patching_plan(PatchRequest(input_cube=(1, hgt, wid), width=256, height=256,
                                                                    stride=128))
    rows = sorted({r for r, _ in plan.patch_coordinates})
    print(f"  {hgt} x {wid}: {len(plan.patch_coordinates)} patches, rows {rows[:3]}...{rows[-3:]} ({len(rows)})")
plan = PatchPlanGenerator().generate_patching_plan(PatchRequest(input_cube=(1, 768, 768), width=256, height=256,
                                                                stride=128))
print("  768 x 768 rows:", sorted([r for r, c in plan.patch_coordinates if c == 0]))

print("== a synthetic test: smooth field + 3x3 hot spot, real weights, two-pass checkerboard")
from app.foundation_models.inferencers.segformer_mae_inferencer import SegFormerMAEInferencer
from app.models.training.inference_config import InferenceConfig
icfg = InferenceConfig(foundation_model_name="segformer_mae", model_config=rm.model_config,
                       checkpoint_path=str(D / "segformer_mae_v0.8.0_epoch495.pt"), patch_size=256, stride=128,
                       inference_batch_size=8, pixel_stats_path=rm.pixel_stats_abs_path, device="cpu")
inf = SegFormerMAEInferencer(icfg)
print("  masking strategy", icfg.masking_strategy, "| erosion", icfg.erosion_kernel_size,
      "| checkerboard cell", icfg.checkerboard_cell_size)
yy, xx = torch.meshgrid(torch.arange(256.0), torch.arange(256.0), indexing="ij")
field = 25.0 + 3.0 * torch.sin(xx / 40.0) + 2.0 * torch.cos(yy / 30.0)
hot = field.clone()
hot[126:129, 126:129] = 45.0                 # 3 x 3, straddles four stage-1 tokens (rows/cols 124-127 and 128-131)
aligned = field.clone()
aligned[124:128, 124:128] = 45.0             # 4 x 4, exactly one stage-1 token
single = field.clone()
single[125, 125] = 45.0                      # one pixel
print("  field at centre", round(field[127, 127].item(), 3), "at (125,125)", round(field[125, 125].item(), 3))
with torch.no_grad():
    for name, f in (("plain", field), ("hot 3x3 straddling", hot), ("hot 4x4 one token", aligned),
                    ("hot 1 pixel", single)):
        rec = inf.predict(f[None, None], torch.ones(1, 1, 256, 256))[0, 0]
        r = (f - rec).abs()
        inner = r[16:-16, 16:-16]
        print(f"  {name}: |res| median {inner.median():.3f}  p99 {inner.flatten().kthvalue(int(0.99*inner.numel())).values:.3f}"
              f"  max {inner.max():.3f}  at hot spot centre {r[127,127]:.3f}  recon at centre {rec[127,127]:.3f}"
              f"  at (125,125) {r[125,125]:.3f} recon {rec[125,125]:.3f}  ring mean residual {r[120:135,120:135].mean():.3f}")
        if name.startswith("hot 3x3"):
            print("   3x3 residuals:", [[round(r[i, j].item(), 2) for j in range(125, 130)] for i in range(125, 130)])

print("== HotSat-like DN field (5000 +/- 400 [docstring figure]) through Chakshu: baked stats vs per-scene override")
from app.models.training.inference_config import PixelStatsOverride
dn = 5000.0 + 400.0 * torch.sin(xx / 40.0) * torch.cos(yy / 30.0)
dn_hot = dn.clone()
dn_hot[124:128, 124:128] = 7000.0
ms, ss = float(dn.mean()), float(dn.std(unbiased=False))
print(f"  field mean {ms:.1f} std {ss:.1f}; z of the mean under baked stats {(ms-mean)/std:.1f}")
for label, kw in (("baked (override never fires)", dict(pixel_stats_path=rm.pixel_stats_abs_path)),
                  ("per-scene override", dict(pixel_stats_path=None,
                                              pixel_stats_override=PixelStatsOverride(mean=[ms], std=[ss],
                                                                                     source="per_scene_dn_zscore")))):
    inf2 = SegFormerMAEInferencer(InferenceConfig(foundation_model_name="segformer_mae", model_config=rm.model_config,
                                                  checkpoint_path=str(D / "segformer_mae_v0.8.0_epoch495.pt"),
                                                  patch_size=256, stride=128, device="cpu", **kw))
    with torch.no_grad():
        for name, f in (("plain", dn), ("hot 4x4 one token", dn_hot)):
            rec = inf2.predict(f[None, None], torch.ones(1, 1, 256, 256))[0, 0]
            r = (f - rec).abs()[16:-16, 16:-16]
            print(f"  {label} / {name}: |res| median {r.median():.1f} DN, max {r.max():.1f}, at hot token "
                  f"{(f - rec).abs()[125, 125]:.1f}, recon mean {rec.mean():.1f}")
print("  buffers after loading, with the override:", inf2.model.normalize.mean.flatten().tolist(),
      inf2.model.normalize.std.flatten().tolist(), "(the override asked for", [round(ms, 1)], [round(ss, 1)], ")")
m3 = SegFormerMAE(in_channels=1, embed_dims=mc["embed_dims"], num_heads=mc["num_heads"],
                  reduction_ratios=mc["reduction_ratios"], num_blocks=mc["num_blocks"], decoder_dim=mc["decoder_dim"],
                  expansion_ratio=mc["expansion_ratio"], pixel_mean=[ms], pixel_std=[ss])
print("  built with override stats:", m3.normalize.mean.item(), "-> after load_state_dict(checkpoint):",
      (m3.load_state_dict(sd) and m3.normalize.mean.item()))

print("== Landsat 9 L2SP B10: DN -> Kelvin -> Celsius -> z (l2sp_dn_to_temperature_transformer.py)")
for dnv in (43500, 52000):
    k = 0.00341802 * dnv + 149.0
    c = k - 273.15
    print(f"  DN {dnv}: {0.00341802*dnv:.6f} + 149 = {k:.6f} K; - 273.15 = {c:.6f} C; z = ({c:.4f} - {mean:.4f})/{std:.4f}"
          f" = {(c-mean)/std:.5f}")
print("== learning rate at epoch 495 of a 500-epoch cosine, 1e-4 -> 1e-6:",
      1e-6 + (1e-4 - 1e-6) * (1 + math.cos(math.pi * 495 / 500)) / 2)
print("== trimming over a batch of 64 fully valid 256 x 256 patches at mask ratio 0.7")
hid = int(4096 * 0.7)
px = hid * 16
print("  hidden tokens", hid, "hidden pixels per patch", px, "per batch", 64 * px,
      "kept by the trim", int(64 * px * 0.95), "dropped", 64 * px - int(64 * px * 0.95))
print("  steps per epoch 30000/64 =", 30000 / 64)
