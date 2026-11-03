"""Loader and encoder sanity checks (README: data)."""
import numpy as np, torch
import common as c

for n in c.DATASETS:
    s = c.load(n)
    print(n, s.Y.shape, s.H, s.W, "K", len(s.E), "Asum", round(float(s.A.sum(0).mean()), 3), round(float(s.A.min()), 3),
          "Y", round(float(s.Y.min()), 3), round(float(s.Y.max()), 3), "E", round(float(s.E.min()), 3), round(float(s.E.max()), 3),
          "wl", s.wl[0], s.wl[-1], s.note, flush=True)
    # spatial check: neighbour correlation of the first band, rows vs columns
    cb = s.cube[..., s.Y.shape[1] // 2]
    print("   nbr corr", round(float(np.corrcoef(cb[:, 1:].ravel(), cb[:, :-1].ravel())[0, 1]), 3),
          round(float(np.corrcoef(s.A.reshape(len(s.E), s.H, s.W)[0][:, 1:].ravel(), s.A.reshape(len(s.E), s.H, s.W)[0][:, :-1].ravel())[0, 1]), 3))
s = c.load("Jasper"); f = c.Frame(s.Y)
X = f(s.Y[:8]); z = c.encode(X, s.wl)
m = c.hypersl()
with torch.no_grad():
    zu = m.encoder_forward(torch.tensor(X, device=c.DEV)[:, None, :], torch.tensor(s.wl, dtype=torch.float32, device=c.DEV)[None].expand(8, -1), 0.0)[0][:, 0].cpu().numpy()
print("upstream vs ours max abs diff", float(np.abs(zu - z).max()), "norm", float(np.linalg.norm(z, axis=1).mean()))
