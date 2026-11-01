"""T6 (and T2/T3 for context): MAE / bias vs Planet basic SR per harmonisation band group, all pixels and land
(NDWI rule), using report.py's own pairing. Plus the climatology AOT windows for the Tanager scenes."""
import sys, glob
sys.path[:0] = ["/content/modH/src", "/content/modH"]
from pathlib import Path
import numpy as np
import report
from harmonise import bands, classes
g = np.load("/content/modH/configs/aod_clim_1deg.npz")
def clim(lat, lon, month):
    a = float(g["aod55"][month - 1, int(np.floor(lat + 90)), int(np.floor(lon + 180)) % 360])
    return round(a, 3), (round(max(0.05, a - 0.15), 2), round(a + 0.15, 2))
print("clim window Permian May", clim(31.66, -102.27, 5), "| ROCX Sep", clim(43.14, -77.50, 9),
      "| Thessaloniki Jul", clim(40.63, 22.96, 7))
T = "/content/data/t11/tan"
for case, sid in (("tan_flare", "20250528_180922_74_4001"), ("tan_rocx", "20250919_170238_88_4001"),
                  ("tan_thess", "20250712_100036_58_4001")):
    prod = report.Product(Path(f"/content/t11/{case}"))
    m = report.match_reference(prod, report.open_reference(f"{T}/{sid}/{sid}_basic_sr_hdf5.h5"))
    o, r, b = m["ours"], m["ref"], m["bands"]
    grp = bands.band_groups(prod.wl)
    land = classes.labels(o, prod.wl) == "land"
    print(f"\n{case}: {m['n']:,} pixel pairs, land {land.mean():.1%}")
    for name in ("core", "edge_lo", "edge_hi", "weak", "strong"):
        sel = b & (grp == name)
        if not sel.any():
            continue
        d_all, d_land = (o - r)[:, sel], (o - r)[land][:, sel]
        print(f"  {name:8s} {sel.sum():3d} bands | all MAE {np.nanmean(abs(d_all)):.4f} bias {np.nanmean(d_all):+.4f}"
              f" | land MAE {np.nanmean(abs(d_land)):.4f} bias {np.nanmean(d_land):+.4f}")
    core = b & (grp == "core")
    for lo, hi in ((460, 600), (600, 1000), (1000, 1800), (1800, 2450)):
        sel = core & (prod.wl >= lo) & (prod.wl < hi)
        d = (o - r)[land][:, sel]
        print(f"    core {lo}-{hi} nm: land MAE {np.nanmean(abs(d)):.4f} bias {np.nanmean(d):+.4f}")
