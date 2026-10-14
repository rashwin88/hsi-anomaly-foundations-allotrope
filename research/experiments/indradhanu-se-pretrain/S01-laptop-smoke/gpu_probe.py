"""Peak memory and step time per stem, batch size and precision, at Indradhanu's real encoder size."""
import sys, time, torch
sys.path.insert(0, ".")
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
wl = torch.linspace(418, 2445, 202).cuda(); st = (wl.tolist(), [0.2] * 202, [0.1] * 202)
real = dict(embed_dims=(32, 64, 160, 256), num_heads=(2, 2, 5, 8), reduction_ratios=(8, 4, 2, 1), num_blocks=(2, 2, 2, 2), decoder_dim=256)
print(f"{'stem':16s} {'prec':5s} {'batch':>5s} {'peak GiB':>9s} {'ms/step':>8s} {'patch/s':>8s}")
for stem in sys.argv[1:] or ("linear", "gate", "cross_attention"):
    m = SpectralStemSegFormerMAE(stem, *st, **real).cuda(); opt = torch.optim.Adam(m.parameters(), 1e-4)
    for prec in ("fp32", "bf16"):
        for b in (1, 2, 4, 8, 16, 32, 64):
            try:
                x = torch.rand(b, 202, 128, 128, device="cuda") * 0.4
                keep, _ = TokenMasking.generate_prediction_mask(torch.ones(b, 1024, device="cuda"), 0.65, "cuda")
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                for i in range(4):
                    if i == 1: torch.cuda.synchronize(); t = time.time()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=prec == "bf16"):
                        loss = (m(x, wl, None, keep) - x).abs().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                torch.cuda.synchronize(); dt = (time.time() - t) / 3
                print(f"{stem:16s} {prec:5s} {b:5d} {torch.cuda.max_memory_allocated() / 2**30:9.2f} {dt * 1000:8.0f} {b / dt:8.1f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{stem:16s} {prec:5s} {b:5d}       OOM", flush=True); break
            finally:
                x = keep = loss = None; torch.cuda.empty_cache()
    del m, opt; torch.cuda.empty_cache()
