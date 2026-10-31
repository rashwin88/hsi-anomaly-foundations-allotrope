"""Item 9 test fixture: give one item 8 case what a full ISOFIT run would leave behind (fake values).

usage: t9_fake.py <work dir> <product dir>
  <work>/isofit/output/<fid>_atm_interp  ENVI, bands AOT550 / H2OSTR / SURFACE_ELEVATION_KM (smooth ramps)
  <work>/stage_times.txt                 read / isofit / post_process / harmonise / write / end
  <product>/<name>_outlier.tif, _snow.tif  a few flagged pixels
  provenance.json gains an atm_post record
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

work, out = Path(sys.argv[1]), Path(sys.argv[2])
fid = (work / "fid.txt").read_text().strip()
name = f"{fid}_L2_SR"
with rasterio.open(out / f"{name}.tif") as d:
    prof = dict(width=d.width, height=d.height, transform=d.transform, crs=d.crs)
    h, w = d.height, d.width
    b0 = d.read(1)
o = work / "isofit" / "output"
o.mkdir(parents=True, exist_ok=True)
yy, xx = np.mgrid[0:h, 0:w] / max(h, w)
bands = [0.15 + 0.2 * xx, 1.2 + 0.8 * yy, 0.3 + 0.1 * (xx + yy)]
with rasterio.open(o / f"{fid}_atm_interp", "w", driver="ENVI", count=3, dtype="float32", **prof) as d:
    for i, (b, n) in enumerate(zip(bands, ("AOT550", "H2OSTR", "SURFACE_ELEVATION_KM"))):
        d.write(np.where(b0 == -9999, -9999, b).astype(np.float32), i + 1)
        d.set_band_description(i + 1, n)
t = 1761900000
(work / "stage_times.txt").write_text("".join(f"{s} {t + dt}\n" for s, dt in
    (("read", 0), ("isofit", 40), ("post_process", 1500), ("harmonise", 1560), ("write", 1580), ("end", 1600))))
rng = np.random.default_rng(1)
for tag, val in (("outlier", 1), ("snow", 1)):
    m = np.zeros((h, w), np.uint8)
    m[b0 == -9999] = 255
    r, c = rng.integers(0, h, 400), rng.integers(0, w, 400)
    m[r, c] = np.where(m[r, c] == 255, 255, val)
    if tag == "snow":
        m[h // 3:h // 3 + 40, w // 3:w // 3 + 40] = 1
    with rasterio.open(out / f"{name}_{tag}.tif", "w", driver="GTiff", count=1, dtype="uint8", nodata=255, **prof) as d:
        d.write(m, 1)
p = out / f"{name}_provenance.json"
prov = json.loads(p.read_text())
prov["atm_post"] = {"aot_prior": {"aod550_clim": 0.31}, "adjacency_applied": True, "terrain_applied": False,
                    "steep_fraction": 0.004, "outlier_pixels_nodata": 1234, "bright_pixels_flagged": 400,
                    "snow_fraction": 0.0015}
p.write_text(json.dumps(prov, indent=1))
print("fixture ready", fid, h, w)
