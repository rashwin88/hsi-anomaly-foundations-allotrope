"""Numbers for the 2026-10-31 additions to chapters 30, 37, 40 and 41 (Heavy in the product, the FT08 leak, the
two-level namer). Builds one model on CPU, no training.
Run from the repo root: C:/av/Scripts/python.exe research/textbook/tools/compute_heavy_product.py
Output: research/textbook/tools/compute_heavy_product_output.txt"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE  # noqa: E402


def n(m):
    return sum(p.numel() for p in m.parameters())


print("== fine-tuned Heavy (FTH3 config.json model_config), parameter count")
wl = [400.0 + 10 * i for i in range(202)]
m = SpectralStemSegFormerMAE("reduced_attention", wl, [0.1] * 202, [0.05] * 202, band_weighting="width",
                             compressed_channels=64, token_dim=32, embed_dims=(96, 192, 480, 768), num_heads=(3, 6, 10, 12),
                             decoder_dim=512, decoder_type="progressive", decoder_stage_dims=(256, 128),
                             seg_classes=5, seg_decoder_dim=256, attention_heads=4, gate_hidden=32, bin_nm=40.0)
print(f"   trainable {n(m):,}")
for k, v in m.named_children():
    print(f"   {k}: {n(v):,}")
sd = m.state_dict()
print(f"   state-dict entries without stats.*: {sum(v.numel() for k, v in sd.items() if not k.startswith('stats')):,}")

print("== parity (FTH3 README)")
prod = [0.2617, 0.1874, 0.0861, 0.2217]
ev = [0.26165, 0.18742, 0.08609, 0.22167]
print(f"   product sum {sum(prod):.4f} mean {np.mean(prod):.5f}; evaluator sum {sum(ev):.5f} mean {np.mean(ev):.5f}")
print("   gaps", [round(a - b, 5) for a, b in zip(prod, ev)])
print(f"   objects 418/909 = {418 / 909:.4f}, 449/909 = {449 / 909:.4f}, gain {449 - 418}")

print("== threads (FTH3 README, cpu_timing.py)")
rows = {"Heavy total": (1819, 454), "Heavy predict": (1629, 402), "Heavy segment": (185, 47), "Heavy score": (4, 4),
        "FT04 total": (1058, 274), "FT04 predict": (920, 236), "FT04 segment": (133, 32)}
for k, (a, b) in rows.items():
    print(f"   {k}: {a} -> {b}, x{a / b:.2f}")
print(f"   minutes Heavy {1819 / 60:.1f} -> {454 / 60:.1f}; FT04 {1058 / 60:.1f} -> {274 / 60:.1f}; Heavy/FT04 at 16: {454 / 274:.2f}")
p = 3 / 3.25
print(f"   Amdahl p from speed-up 4 between 2 and 16 threads: {p:.4f}")
for nt in (16, 24):
    s = ((1 - p) + p / 2) / ((1 - p) + p / nt)
    print(f"   predicted speed-up 2 -> {nt}: {s:.2f}")
# exact p from the measured 4.007: (1 - p/2) = S (1 - 15p/16)  ->  p = (S - 1) / (15S/16 - 1/2)
S = 1819 / 454
print(f"   p from measured {S:.3f}: {(S - 1) / (15 * S / 16 - 0.5):.4f}")

print("== FT08 leak on a 4x4 / 6x6 token tile")
for side in (4, 6):
    i, j = np.indices((side, side))
    h = ((i // 2 + j // 2) % 2 == 1).astype(int)
    s = ((i + j) % 2 == 1).astype(int)
    print(f"   {side}x{side}: leaks {(h != s).sum()} of {side * side}")
    if side == 4:
        print("   h\n", h, "\n   s\n", s, "\n   leak\n", (h != s).astype(int))

print("== separability (read only)")
for sensor in ("prisma", "enmap", "aviris_ng"):
    f = Path(f"research/experiments/spectral-namer/separability_{sensor}.json")
    d = json.loads(f.read_text())
    for fill, v in d.items():
        print(f"   {sensor} fill {fill}: n {v['n']}, L1 mm {v['level1_manmade_called_manmade']:.4f}, "
              f"nat {v['level1_natural_called_natural']:.4f}, L2 {v['level2_correct_family']:.4f}, "
              f"L2|L1 {v['level2_correct_given_level1_right']:.4f}, "
              f"product {v['level1_manmade_called_manmade'] * v['level2_correct_given_level1_right']:.4f}")
