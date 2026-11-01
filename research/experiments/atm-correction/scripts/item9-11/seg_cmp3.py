"""AVIRIS-NG 600-line window: seg 50 (clean rerun) / 100 / 200 vs JPL corr per band group (land), vs seg 50 directly,
retrieved atmosphere, and ISOFIT stage times from the logs."""
import sys, glob, re
sys.path[:0] = ["/content/modT/src", "/content/modT"]
from pathlib import Path
import numpy as np, rasterio
import report
from harmonise import bands, classes
REF = "/content/data/t11/ang600/l2/ang20151219t081738_corr_v2m2_img.hdr"
CASES = ("ang600_s50b", "ang600_s100", "ang600_s200")
P = {c: report.Product(Path(f"/content/t11/{c}")) for c in CASES}
for c in CASES:
    p = P[c]
    m = report.match_reference(p, report.open_reference(REF))
    o, r, b = m["ours"], m["ref"], m["bands"]
    grp = bands.band_groups(p.wl)
    land = classes.labels(o, p.wl) == "land"
    log = open(f"/content/t11/{c}.log").read()
    ts = re.findall(r"INFO:2026-\d\d-\d\d,(\d\d:\d\d:\d\d) \|\| \S+ \| (Beginning \d+ inversions|Pixel class: uniform_surface inversions complete|Analytical line inversions complete)", log)
    st = [l.split() for l in open(f"/content/t11/{c}/_work/stage_times.txt")]
    print(f"\n{c}: total {(int(st[-1][1]) - int(st[0][1])) / 60:.1f} min | " + " ; ".join(f"{t} {s[:28]}" for t, s in ts))
    for g in ("core", "edge_lo", "edge_hi", "weak", "strong"):
        sel = b & (grp == g)
        if sel.any():
            d = (o - r)[land][:, sel]
            print(f"  {g:8s} {sel.sum():3d} bands | land MAE {np.nanmean(abs(d)):.4f} bias {np.nanmean(d):+.4f}")
a = P["ang600_s50b"]
grp = bands.band_groups(a.wl)
for c in CASES[1:]:
    ok = a.valid & P[c].valid
    d = (P[c].rfl - a.rfl)[ok]
    print(f"\n{c} - seg50 on {ok.sum():,} px: " + " | ".join(
        f"{g} mean {np.nanmean(abs(d[:, grp == g])):.4f} p99 {np.nanpercentile(abs(d[:, grp == g]), 99):.4f}"
        for g in ("core", "edge_lo", "weak", "strong") if (grp == g).any()))
for c in CASES:
    f = [q for q in glob.glob(f"/content/t11/{c}/_work/isofit/output/*_atm_interp") if "_subs_" not in q][0]
    with rasterio.open(f) as s:
        names = [(n or "").upper() for n in s.descriptions]
        out = []
        for k in ("AOT550", "H2OSTR"):
            v = s.read(names.index(k) + 1)[::5, ::5]
            v = v[v > -9990]
            out.append(f"{k} {np.median(v):.3f} ({np.percentile(v, 5):.3f}-{np.percentile(v, 95):.3f})")
        print(f"{c:12s} " + " | ".join(out))
