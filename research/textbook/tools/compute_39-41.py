"""Parameter counts for chapters 39-41 (uncertainty, decoderwork, heavy), built on CPU, no training.
Run from the repo root: C:/av/Scripts/python.exe research/textbook/tools/compute_39-41.py"""
import sys
sys.path.insert(0, ".")
from app.foundation_models.components.seg_former_decoder import SegFormerDecoder
from app.foundation_models.components.progressive_decoder import ProgressiveDecoder
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE

def n(m):
    return sum(p.numel() for p in m.parameters())

print("uncertainty head SegFormerDecoder([32,64,160,256], 64, 1):", n(SegFormerDecoder([32, 64, 160, 256], decoder_dim=64, out_channels=1)))
print("same, out_channels=2:", n(SegFormerDecoder([32, 64, 160, 256], decoder_dim=64, out_channels=2)))
print("FT09 ProgressiveDecoder FT04 widths:", n(ProgressiveDecoder([32, 64, 160, 256])))
print("FT04 SegFormerDecoder (256, 32*1 out):", n(SegFormerDecoder([32, 64, 160, 256], decoder_dim=256, out_channels=32)))
wl = [400.0 + 10 * i for i in range(202)]
mean, std = [0.1] * 202, [0.05] * 202
def build(**kw):
    return SpectralStemSegFormerMAE("reduced_attention", wl, mean, std, band_weighting="width", **kw)
heavy = build(compressed_channels=64, token_dim=32, embed_dims=(96, 192, 480, 768), num_heads=(3, 6, 10, 12),
              decoder_dim=512, decoder_type="progressive", decoder_stage_dims=(256, 128))
base = build(compressed_channels=32, num_heads=(2, 2, 5, 8))
for name, m in (("Heavy", heavy), ("FT04 size (no seg head)", base)):
    print(f"== {name}: total {n(m):,}")
    for child_name, child in m.named_children():
        print(f"   {child_name}: {n(child):,}")
    enc = m.encoder
    print("   encoder.patch_embeds[0]:", f"{n(enc.patch_embeds[0]):,}", " patch_embeds[1]:", f"{n(enc.patch_embeds[1]):,}")
    print("   encoder.blocks[0]:", f"{n(enc.blocks[0]):,}", " one block:", f"{n(enc.blocks[0][0]):,}")
print("== per stage (one block / patch embed / final norm if any), Heavy vs FT04 size")
for i in range(4):
    hb, bb = n(heavy.encoder.blocks[i][0]), n(base.encoder.blocks[i][0])
    hp, bp = n(heavy.encoder.patch_embeds[i]), n(base.encoder.patch_embeds[i])
    print(f"   stage {i+1}: block Heavy {hb:,} FT04 {bb:,} ratio {hb/bb:.2f}; patch embed Heavy {hp:,} FT04 {bp:,}")
rest_h = n(heavy.encoder) - sum(n(heavy.encoder.blocks[i]) + n(heavy.encoder.patch_embeds[i]) for i in range(4))
rest_b = n(base.encoder) - sum(n(base.encoder.blocks[i]) + n(base.encoder.patch_embeds[i]) for i in range(4))
print("   encoder remainder (stage norms): Heavy", rest_h, "FT04", rest_b)
print("== Heavy decoder children:", {k: n(v) for k, v in heavy.decoder.named_children()})
print("== F002-heavy metrics.csv checks")
import csv, math
rows = list(csv.DictReader(open("research/experiments/indradhanu-se-pretrain/F002-heavy/metrics.csv")))
tr = [float(r["train_l1"]) for r in rows]; va = [float(r["val_l1"]) for r in rows]
gaps = [tr[i] - va[i] for i in range(19, 80)]
print(f"   train - val L1 gap, epochs 20-80: min {min(gaps):.5f} max {max(gaps):.5f}")
jumps = [(i + 1, tr[i] - tr[i - 1]) for i in range(5, 80, 5)]   # first epoch of each new block (6, 11, ...)
print("   largest train-L1 rise at a block change: epoch %d %+.5f" % max(jumps, key=lambda j: j[1]))
print("   total seconds:", round(sum(float(r["seconds"]) for r in rows), 1))
for e in (1, 40, 60, 80):
    print(f"   lr epoch {e}: csv {float(rows[e-1]['lr']):.4e}  formula {1e-6 + (3e-4 - 1e-6) * (1 + math.cos(math.pi * e / 80)) / 2:.4e}")
tf = sum(tr[b * 5 + 4] < tr[b * 5] for b in range(16)); vf = sum(va[b * 5 + 4] < va[b * 5] for b in range(16))
print(f"   blocks (5 epochs) whose last epoch is below its first: train {tf} of 16, val {vf} of 16")
