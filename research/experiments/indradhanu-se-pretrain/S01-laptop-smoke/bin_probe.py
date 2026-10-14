"""Prototype: cross-attention over fixed wavelength bins. Bin token = masked mean of band tokens,
computed from the bin's mean value and mean wavelength tag (value embedding is linear)."""
import sys, time, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, ".")
import app.foundation_models.components.spectral_attention as sa
from app.foundation_models.components.spectral_tokens import wavelength_encoding
from app.foundation_models.components.spectral_stem_seg_former_mae import SpectralStemSegFormerMAE
from app.foundation_models.components.token_masking import TokenMasking
from app.models.dataset.spectralearth import load_band_table

class BinnedCompressor(sa.SpectralCrossAttentionCompressor):
    def __init__(self, bin_nm=40.0, **kw):
        super().__init__(**kw); self.bin_nm = bin_nm
    def forward(self, x, wl, band_valid=None):
        b, n, h, w = x.shape; p, hd = b * h * w, self.d // self.heads
        idx = ((wl - 400.0) // self.bin_nm).long(); used, idx = torch.unique(idx, return_inverse=True); g = len(used)
        M = F.one_hot(idx, g).to(x.dtype)                                         # (N, G)
        rows = x.permute(0, 2, 3, 1).reshape(p, n)
        valid = torch.ones_like(rows) if band_valid is None else band_valid.permute(0, 2, 3, 1).reshape(p, n).to(x.dtype)
        cnt = valid @ M                                                           # (P, G)
        vmean = (rows * valid) @ M / cnt.clamp(min=1)                             # (P, G)
        enc = wavelength_encoding(wl, self.d).to(x.dtype)                         # (N, d)
        emean = (valid @ (M[:, :, None] * enc[:, None, :]).reshape(n, g * self.d)).view(p, g, self.d) / cnt.clamp(min=1)[..., None]
        tokens = self.value_embed(vmean[..., None]) + emean                       # (P, G, d)
        k = self.k(tokens).view(p, g, self.heads, hd).transpose(1, 2); v = self.v(tokens).view(p, g, self.heads, hd).transpose(1, 2)
        q = self.q(self.queries).view(self.D, self.heads, hd).transpose(0, 1).unsqueeze(0).expand(p, -1, -1, -1)
        mask = (cnt > 0).view(p, 1, 1, g)
        out = sa._attend(q, k, v, mask)
        out = self.to_channel(self.o(out.transpose(1, 2).reshape(p, self.D, self.d)))
        return out.view(b, h, w, self.D).permute(0, 3, 1, 2)

wl = torch.tensor(load_band_table().wavelengths_nm, dtype=torch.float32).cuda(); st = (wl.tolist(), [0.2] * 202, [0.1] * 202)
real = dict(embed_dims=(32, 64, 160, 256), num_heads=(2, 2, 5, 8), reduction_ratios=(8, 4, 2, 1), num_blocks=(2, 2, 2, 2), decoder_dim=256)
for bin_nm in [float(a) for a in sys.argv[1:]] or [20.0, 40.0, 80.0]:
    m = SpectralStemSegFormerMAE("cross_attention", *st, **real)
    m.spectral_in = BinnedCompressor(bin_nm=bin_nm); m = m.cuda(); opt = torch.optim.Adam(m.parameters(), 1e-4)
    g = len(torch.unique(((wl - 400) // bin_nm).long()))
    for bs in (4, 8, 16, 32):
        try:
            x = torch.rand(bs, 202, 128, 128, device="cuda") * 0.4
            keep, _ = TokenMasking.generate_prediction_mask(torch.ones(bs, 1024, device="cuda"), 0.65, "cuda")
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            for i in range(4):
                if i == 1: torch.cuda.synchronize(); t = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = (m(x, wl, None, keep) - x).abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            torch.cuda.synchronize(); dt = (time.time() - t) / 3
            print(f"bin {bin_nm:4.0f} nm  tokens {g:3d}  batch {bs:3d}  peak {torch.cuda.max_memory_allocated() / 2**30:6.2f} GiB  {bs / dt:6.1f} patch/s", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"bin {bin_nm:4.0f} nm  batch {bs}: OOM"); break
        finally:
            x = keep = loss = None; torch.cuda.empty_cache()
    del m, opt; torch.cuda.empty_cache()
