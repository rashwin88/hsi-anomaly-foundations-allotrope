"""GT-HAD (Lian et al., TNNLS 2025) on full PRISMA L2D scenes.

The network (net.py) is imported unchanged from the official repo
(github.com/jeline0110/GT-HAD). Hyper-parameters follow the paper / official main.py:
9x9 cubes, stride 3, embed 64, mlp_ratio 2, Adam lr 1e-3, batch 64, 150 epochs,
CMM every 25 epochs, AvgPool3d(5,3,3) on the residual, overlap-averaged fold.

What differs from the official main.py (memory/speed only, same maths):
  * cubes are cut on the fly from a replicate-padded image instead of being
    materialised (N x D x 9 x 9 is ~12 GB at PRISMA size);
  * fold is an index_add_ into an image buffer instead of F.fold over all N cubes;
  * CMM nearest-neighbour search uses chunked |k|^2+|q|^2-2kq matmuls (fp32, TF32 off)
    instead of a python loop of PairwiseDistance rows.

Usage: python gthad_prisma.py <set_id> <out_dir> [--epochs 150]
"""
import argparse, glob, json, os, sys, time
import h5py, numpy as np, rasterio, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, "/content/GT-HAD/dnnmethods/GT-HAD")
from net import Net  # official network, unchanged

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DATA = "/content/drive/MyDrive/PS_11/stage2prep_holdout/stage2_dataset_extracted/Phase II datasets"
K, S = 9, 3  # cube size, stride


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_prisma(set_id):
    he = glob.glob(f"{DATA}/Set-{set_id}/*/*.he5")[0]
    gt_path = [g for g in glob.glob(f"{DATA}/Set-{set_id}/*/Ground*/*.tif")
               if "hyp" in g.lower()][0]
    with h5py.File(he, "r") as f:
        a = f.attrs
        fld = f["HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields"]
        cubes = []
        for name, lo, hi in [("VNIR_Cube", "L2ScaleVnirMin", "L2ScaleVnirMax"),
                             ("SWIR_Cube", "L2ScaleSwirMin", "L2ScaleSwirMax")]:
            dn = fld[name][()].astype(np.float32)          # (H, B, W)
            refl = a[lo] + dn * (a[hi] - a[lo]) / 65535.0   # PRISMA L2 DN -> reflectance
            cubes.append(refl.transpose(1, 0, 2))           # (B, H, W)
        meta = {k: (a[k].tolist() if hasattr(a[k], "tolist") else str(a[k]))
                for k in a.keys() if "corner" in k.lower() or k == "Epsg_Code"}
    x = np.concatenate(cubes, 0)
    with rasterio.open(gt_path) as r:
        gt = (r.read(1) > 0).astype(np.uint8)
        gt_tf = tuple(r.transform)[:6]
    return x, gt, dict(he5=he, gt=gt_path, gt_transform=gt_tf, he5_attrs=meta)


class Cubes:
    """On-the-fly 9x9 cube extraction / fold for one (D,H,W) image on the GPU."""

    def __init__(self, img):  # img: (D,H,W) cuda
        D, H, W = img.shape
        self.D, self.H, self.W = D, H, W
        orows, ocols = -(-H // S), -(-W // S)
        pr, pc = max(0, (orows - 1) * S + K - H), max(0, (ocols - 1) * S + K - W)
        self.pt, self.pl = pr // 2, pc // 2
        self.pad = (self.pl, pc - self.pl, self.pt, pr - self.pt)
        self.Hp, self.Wp = H + pr, W + pc
        self.nr, self.nc = (self.Hp - K) // S + 1, (self.Wp - K) // S + 1
        self.N = self.nr * self.nc
        dy, dx = torch.meshgrid(torch.arange(K), torch.arange(K), indexing="ij")
        self.off = (dy * self.Wp + dx).flatten().cuda()  # (81,)
        self.set_image(img)

    def padded(self, img):
        return nn.functional.pad(img[None], self.pad, mode="replicate")[0]

    def set_image(self, img):
        self.img = img
        self.view = self.padded(img).unfold(1, K, S).unfold(2, K, S)  # D,nr,nc,K,K

    def get(self, idx, view=None):  # -> (n,D,K,K)
        v = self.view if view is None else view
        return v[:, idx // self.nc, idx % self.nc].permute(1, 0, 2, 3).contiguous()

    def base(self, idx):
        return (idx // self.nc) * S * self.Wp + (idx % self.nc) * S

    def fold_add(self, buf, cnt, idx, cubes):  # buf (D,Hp*Wp), cubes (n,D,K,K)
        flat = (self.base(idx)[:, None] + self.off[None]).flatten()
        buf.index_add_(1, flat, cubes.reshape(len(idx), self.D, K * K)
                       .permute(1, 0, 2).reshape(self.D, -1))
        cnt.index_add_(0, flat, torch.ones_like(flat, dtype=cnt.dtype))

    def fold_finish(self, buf, cnt):
        out = (buf / cnt.clamp_min(1)).view(self.D, self.Hp, self.Wp)
        return out[:, self.pt:self.pt + self.H, self.pl:self.pl + self.W]


def cmm(cb, recon_img, match_vec, kchunk=2048, qchunk=8192):
    """Content-matching: block i -> BFB iff argmin_j ||recon_i - orig_j|| == i."""
    rview = cb.padded(recon_img).unfold(1, K, S).unfold(2, K, S)
    allidx = torch.arange(cb.N, device="cuda")
    qn = torch.empty(cb.N, device="cuda")
    for q0 in range(0, cb.N, qchunk):
        q = cb.get(allidx[q0:q0 + qchunk]).flatten(1)
        qn[q0:q0 + qchunk] = (q * q).sum(1)
    for k0 in range(0, cb.N, kchunk):
        kidx = allidx[k0:k0 + kchunk]
        k = cb.get(kidx, rview).flatten(1)
        kn = (k * k).sum(1, keepdim=True)
        best = torch.full((len(kidx),), float("inf"), device="cuda")
        arg = torch.zeros(len(kidx), dtype=torch.long, device="cuda")
        for q0 in range(0, cb.N, qchunk):
            q = cb.get(allidx[q0:q0 + qchunk]).flatten(1)
            d = kn + qn[q0:q0 + qchunk][None] - 2 * k @ q.T
            m, a = d.min(1)
            upd = m < best
            best[upd], arg[upd] = m[upd], a[upd] + q0
        match_vec[kidx[arg == kidx]] = 1
    return match_vec


def rx_baseline(x, valid, n_pc=30):
    """Global RX on top PCA components (reference only, not part of GT-HAD)."""
    X = torch.from_numpy(x.reshape(x.shape[0], -1).T).cuda().double()
    v = torch.from_numpy(valid.flatten()).cuda()
    mu = X[v].mean(0)
    Xc = X - mu
    C = Xc[v].T @ Xc[v] / v.sum()
    _, U = torch.linalg.eigh(C)
    Z = Xc @ U[:, -n_pc:]
    Ci = torch.linalg.inv(Z[v].T @ Z[v] / v.sum())
    return ((Z @ Ci) * Z).sum(1).view(x.shape[1:]).float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_id", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search_iter", type=int, default=25)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    t0 = time.time()
    x, gt, info = load_prisma(args.set_id)
    log("loaded", x.shape, "gt", gt.shape, "anomaly px", int(gt.sum()), info)
    assert x.shape[1:] == gt.shape, "GT/cube grid mismatch"
    valid = (x != 0).any(0)                       # L2D fill outside the swath
    keep = x[:, valid].std(1) > 1e-6              # drop dead bands
    x = x[keep]
    log(f"bands kept {keep.sum()}/{keep.size}; valid px {valid.mean():.3f}; "
        f"anomalies inside valid {int(gt[valid].sum())}/{int(gt.sum())}")
    x = (x - x.min()) / (x.max() - x.min())       # official: global min-max to [0,1]

    img = torch.from_numpy(x).cuda()
    D = img.shape[0]
    cb = Cubes(img)
    log(f"cubes N={cb.N} ({cb.nr}x{cb.nc}), padded {cb.Hp}x{cb.Wp}")

    net = Net(in_chans=D, embed_dim=64, patch_size=3, patch_stride=3,
              mlp_ratio=2.0, attn_drop=0.0, drop=0.0).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    match_vec = torch.zeros(cb.N, device="cuda")
    end_iter, search_iter, bs = args.epochs, args.search_iter, 64
    hist = []
    for ep in range(1, end_iter + 1):
        search = ep % search_iter == 0 and ep != end_iter
        if search:
            sbuf = torch.zeros(D, cb.Hp * cb.Wp, device="cuda")
            scnt = torch.zeros(cb.Hp * cb.Wp, device="cuda")
        perm = torch.randperm(cb.N, device="cuda")
        if cb.N % bs == 1:   # official squeeze() in calculate_mask breaks on a size-1 batch
            perm = perm[:-1]
        tot, te = 0.0, time.time()
        for b0 in range(0, len(perm), bs):
            idx = perm[b0:b0 + bs]
            inp = cb.get(idx)
            out = net(inp, block_idx=idx, match_vec=match_vec)
            if search:
                cb.fold_add(sbuf, scnt, idx, out.detach())
            loss = mse(out, inp)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        msg = f"epoch {ep}/{end_iter} loss {tot / len(perm):.3e} ({time.time() - te:.0f}s)"
        if search:
            ts = time.time()
            match_vec = torch.zeros(cb.N, device="cuda")
            match_vec = cmm(cb, cb.fold_finish(sbuf, scnt), match_vec)
            msg += f" | CMM: {int(match_vec.sum())}/{cb.N} blocks -> BFB ({time.time() - ts:.0f}s)"
            del sbuf, scnt
        log(msg)
        hist.append(dict(epoch=ep, loss=tot / len(perm), bfb=int(match_vec.sum())))

    # detection (official: |in-out|^2 -> AvgPool3d(5,3,3) -> fold-average -> sum bands)
    net.eval()
    pool = nn.AvgPool3d(kernel_size=(5, 3, 3), stride=1, padding=(2, 1, 1))
    rbuf = torch.zeros(D, cb.Hp * cb.Wp, device="cuda")
    rcnt = torch.zeros(cb.Hp * cb.Wp, device="cuda")
    with torch.no_grad():
        allidx = torch.arange(cb.N, device="cuda")
        for b0 in range(0, cb.N, 256):
            idx = allidx[b0:b0 + 256]
            if len(idx) == 1:
                idx = allidx[b0 - 1:b0 + 1]
            inp = cb.get(idx)
            res = pool((inp - net(inp, block_idx=idx, match_vec=match_vec)) ** 2)
            cb.fold_add(rbuf, rcnt, idx, res)
    amap = cb.fold_finish(rbuf, rcnt).sum(0)
    amap = ((amap - amap.min()) / (amap.max() - amap.min())).cpu().numpy()
    train_s = time.time() - t0

    rx = rx_baseline(x, valid)
    m = dict(set=args.set_id, bands=int(D), shape=list(gt.shape), n_cubes=cb.N,
             anomaly_px=int(gt.sum()), valid_frac=float(valid.mean()),
             auc_full=float(roc_auc_score(gt.ravel(), amap.ravel())),
             auc_valid=float(roc_auc_score(gt[valid], amap[valid])),
             rx_pca30_auc_valid=float(roc_auc_score(gt[valid], rx[valid])),
             runtime_s=round(train_s), epochs=end_iter, search_iter=search_iter, seed=args.seed,
             final_bfb_blocks=int(match_vec.sum()), info=info)
    # simple operating points on valid pixels
    fpr, tpr, _ = roc_curve(gt[valid], amap[valid])
    for f in (1e-3, 1e-2):
        m[f"pd_at_pf_{f:g}"] = float(np.interp(f, fpr, tpr))
    log("RESULT", json.dumps({k: v for k, v in m.items() if k != "info"}))

    np.save(f"{args.out_dir}/gthad_map.npy", amap.astype(np.float32))
    np.save(f"{args.out_dir}/rx_pca30_map.npy", rx.astype(np.float32))
    np.save(f"{args.out_dir}/roc_valid.npy", np.stack([fpr, tpr]))
    json.dump(m, open(f"{args.out_dir}/metrics.json", "w"), indent=2)
    json.dump(hist, open(f"{args.out_dir}/train_history.json", "w"))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(22, 6))
    rgb = x[[int(D * .45), int(D * .25), int(D * .1)]].transpose(1, 2, 0)
    ax[0].imshow(np.clip(rgb / np.percentile(rgb[valid], 99), 0, 1)); ax[0].set_title(f"Set-{args.set_id} false colour")
    ax[1].imshow(gt, cmap="gray"); ax[1].set_title(f"GT ({int(gt.sum())} px)")
    ax[2].imshow(amap, cmap="jet", vmax=np.percentile(amap[valid], 99.9)); ax[2].set_title(f"GT-HAD map, AUC {m['auc_valid']:.4f}")
    ax[3].plot(fpr, tpr, label=f"GT-HAD {m['auc_valid']:.4f}")
    f2, t2, _ = roc_curve(gt[valid], rx[valid]); ax[3].plot(f2, t2, label=f"RX-PCA30 {m['rx_pca30_auc_valid']:.4f}")
    ax[3].set_xscale("log"); ax[3].set_xlim(1e-5, 1); ax[3].set_xlabel("PF"); ax[3].set_ylabel("PD"); ax[3].legend(); ax[3].set_title("ROC (valid px)")
    for a in ax[:3]: a.axis("off")
    plt.tight_layout(); plt.savefig(f"{args.out_dir}/summary.png", dpi=110)


if __name__ == "__main__":
    main()
