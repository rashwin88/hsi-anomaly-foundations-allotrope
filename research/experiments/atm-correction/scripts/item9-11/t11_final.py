"""Item 11 final table: every trial's report regenerated with the fixed reference pairing, plus land MAE per band
group against its vendor reference (same pairing as the report)."""
import sys, glob, time
sys.path[:0] = ["/content/modT/src", "/content/modT"]
from pathlib import Path
import numpy as np
import report
from harmonise import bands, classes
T = "/content/data/t11"
CASES = [("prs_l4", glob.glob(f"{T}/prs_train/PRS_L2C*.he5")[0], "ASI L2C"),
         ("tan_thess", f"{T}/tan/20250712_100036_58_4001/20250712_100036_58_4001_basic_sr_hdf5.h5", "Planet SR"),
         ("tan_rocx", f"{T}/tan/20250919_170238_88_4001/20250919_170238_88_4001_basic_sr_hdf5.h5", "Planet SR"),
         ("prs_snow", glob.glob(f"{T}/prs_snow/PRS_L2D*.he5")[0], "ASI L2D"),
         ("en_coast", glob.glob(f"{T}/enmap/ENMAP01-____L2A-*")[0], "DLR L2A"),
         ("tan_flare", f"{T}/tan/20250528_180922_74_4001/20250528_180922_74_4001_basic_sr_hdf5.h5", "Planet SR"),
         ("ang", f"{T}/ang/l2/ang20151219t081738_corr_v2m2_img.hdr", "JPL corr")]
for case, ref, name in CASES:
    out, work = Path(f"/content/t11/{case}"), Path(f"/content/t11/{case}/_work")
    t0 = time.time()
    report.build(out, work, ref)
    p = report.Product(out)
    m = report.match_reference(p, report.open_reference(ref))
    o, r, b = m["ours"], m["ref"], m["bands"]
    grp = bands.band_groups(p.wl)
    land = classes.labels(o, p.wl) == "land"
    g = {}
    for gname in ("core", "edge_lo", "edge_hi", "weak", "strong"):
        sel = b & (grp == gname)
        g[gname] = np.nanmean(abs((o - r)[land][:, sel])) if sel.any() else np.nan
    dall = (o - r)[:, b]
    print(f"{case:9s} vs {name:9s} step {p.step} | all-band MAE {np.nanmean(abs(dall)):.4f} bias {np.nanmean(dall):+.4f} | "
          f"land core {g['core']:.4f} (bias {np.nanmean((o - r)[land][:, b & (grp == 'core')]):+.4f}) edge_lo {g['edge_lo']:.4f} "
          f"edge_hi {g['edge_hi']:.4f} weak {g['weak']:.4f} strong {g['strong']:.4f} | {time.time() - t0:.0f} s", flush=True)
print("FINAL_DONE")
