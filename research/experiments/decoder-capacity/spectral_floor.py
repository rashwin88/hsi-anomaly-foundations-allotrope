"""How much of FT04's residual is the spectral floor - the part of each spectrum outside the 32-curve span its linear
output layer can produce (out = U(wl) z + c(wl), normalised units)? Since every reconstruction lies in that span, the
out-of-span part of the residual equals the out-of-span part of the data itself.

    share = sum ||floor residual||^2 / sum ||actual residual||^2, over background and over target pixels.
Also: mean |residual| per band, normalised units. PRISMA bench, no training."""
import sys
import numpy as np, torch
from pathlib import Path
from app.evaluation.config import ModelSpec
from app.evaluation.datasets import PrismaBench
from app.evaluation.models import EvalModel
from app.foundation_models.components.spectral_reduced import wavelength_encoding

m = EvalModel(ModelSpec(name="FT04", checkpoint="C:/es/models/FT04_best.pt"), Path("C:/es/eval/cache_b"))
for sc in PrismaBench("C:/Users/FITT/Documents/stage1_datasets", sys.argv[1:] or ["set1", "set2", "set3", "set4"]):
    cube, recon, valid, _ = m.reconstruct(sc)
    net = m.inferencer.model.inner
    wl = torch.as_tensor(net.wavelengths_nm if hasattr(net, "wavelengths_nm") else m.inferencer.model.wavelengths_nm).float().cpu()
    so = net.spectral_out.cpu()
    with torch.no_grad():
        uc = so.out_weight(wavelength_encoding(wl, so.enc_dim)).numpy().astype(np.float64)
    so.to(m.inferencer.device)
    U, c = uc[:, :so.D], uc[:, so.D]                                   # (N, 32), (N,)
    mean, std = (t.cpu().numpy().astype(np.float64) for t in net.stats.at(wl.to(net.stats.wl.device)))
    norm = lambda a: (a.astype(np.float64) - mean[:, None]) / std[:, None]          # a: (N, pixels)
    Q, _ = np.linalg.qr(U)                                             # orthonormal basis of the span
    t = (sc.objects > 0) & valid
    b = valid & ~t
    rng = np.random.default_rng(0)
    bi = rng.choice(np.flatnonzero(b.ravel()), size=min(200_000, b.sum()), replace=False)
    out = []
    for name, idx in (("background", bi), ("targets", np.flatnonzero(t.ravel()))):
        x = norm(cube.reshape(len(wl), -1)[:, idx]) - c[:, None]
        ra = norm(recon.reshape(len(wl), -1)[:, idx]) - c[:, None] - x   # actual residual (normalised)
        rf = x - Q @ (Q.T @ x)                                         # floor: x minus its projection onto the span
        out.append(f"{name}: |r| actual {np.abs(ra).mean():.3f} floor {np.abs(rf).mean():.3f}  "
                   f"floor share of residual energy {(rf ** 2).sum() / (ra ** 2).sum():.2f}")
    print(sc.name, f"bands {len(wl)} |", " | ".join(out), flush=True)
