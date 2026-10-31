"""Item 11: retrieved AOT550 / water vapour at an AERONET station vs the station (and vs Planet's SR-product layers).

usage: t11_aeronet.py <case> <work dir> <site> <planet sr .h5> [<window radius m>=500]
AERONET: Level 1.5 AOD (version 3, all points) within +-30 min of the overpass; AOD550 from AOD500 and the
440-870 nm Angstrom exponent; precipitable water (cm = g/cm2). Ours: ISOFIT *_atm_interp (AOT550, H2OSTR) within
the radius of the station (nearest pixel by the loc cube), median and 5-95 %. Planet: any Data Fields layer whose
name mentions aerosol / aot / aod / water vapo(u)r, same window.
"""
import csv, io, json, sys, urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import rasterio
from spectral.io import envi

case, work, site, sr = sys.argv[1:5]
rad = float(sys.argv[5]) if len(sys.argv) > 5 else 500.0
work = Path(work)
scene = json.loads((work / "scene.json").read_text())
fid = scene["fid"]
loc = envi.open(str(work / f"{fid}_loc.hdr"), str(work / f"{fid}_loc")).open_memmap(interleave="bip")
lon, lat = np.asarray(loc[..., 0], float), np.asarray(loc[..., 1], float)

sites = urllib.request.urlopen("https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt", timeout=60).read().decode()
row = next(l.split(",") for l in sites.splitlines() if l.startswith(site + ","))
slon, slat = float(row[1]), float(row[2])
k = 111320.0
d = np.hypot((lon - slon) * k * np.cos(np.radians(slat)), (lat - slat) * k)
win = d <= rad
r0, c0 = np.unravel_index(np.nanargmin(d), d.shape)
out = {"case": case, "site": site, "site_latlon": [slat, slon], "nearest_pixel_m": round(float(np.nanmin(d)), 1),
       "pixel_rc": [int(r0), int(c0)], "window_px": int(win.sum()), "radius_m": rad}

atm = [p for p in (work / "isofit" / "output").glob("*_atm_interp") if "_subs_" not in p.name][0]
with rasterio.open(atm) as a:
    names = [(n or "").upper() for n in a.descriptions]
    for key in ("AOT550", "H2OSTR"):
        v = a.read(names.index(key) + 1)[win]
        v = v[v > -9990]
        out[f"ours_{key}"] = [round(float(np.median(v)), 4), round(float(np.percentile(v, 5)), 4), round(float(np.percentile(v, 95)), 4)]

with h5py.File(sr, "r") as f:
    found = {}
    def walk(n, o):
        if isinstance(o, h5py.Dataset) and "Data Fields" in n and o.ndim == 2 and any(
                s in n.lower() for s in ("aerosol", "aot", "aod", "water_vap", "water vap", "wv", "h2o")):
            v = np.asarray(o[...], float)
            if v.shape == win.shape:
                x = v[win]; x = x[np.isfinite(x) & (x > -9990)]
                if x.size:
                    found[n.split("/")[-1]] = [round(float(np.median(x)), 4), round(float(np.percentile(x, 5)), 4),
                                               round(float(np.percentile(x, 95)), 4)]
            else:
                found[n.split("/")[-1]] = f"shape {v.shape} != {win.shape}"
    f.visititems(walk)
out["planet"] = found

stac = json.load(open(next(Path(sr).parent.glob("*_stac_item.json"))))
t = datetime.fromisoformat(stac["properties"]["datetime"].replace("Z", "+00:00")).replace(tzinfo=None)
out["overpass_utc"] = t.isoformat(timespec="seconds")
url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site={site}&year={t.year}&month={t.month}&day={t.day}"
       f"&year2={t.year}&month2={t.month}&day2={t.day}&AOD15=1&AVG=10&if_no_html=1")
body = urllib.request.urlopen(url, timeout=60).read().decode(errors="ignore")
lines = body.splitlines()
h = next(i for i, l in enumerate(lines) if l.startswith("AERONET_Site"))
rows = list(csv.DictReader(io.StringIO("\n".join(lines[h:]))))
sel = []
for r in rows:
    tt = datetime.strptime(r["Date(dd:mm:yyyy)"] + " " + r["Time(hh:mm:ss)"], "%d:%m:%Y %H:%M:%S")
    if abs(tt - t) <= timedelta(minutes=30):
        sel.append(r)
num = lambda r, c: float(r[c]) if r.get(c) not in (None, "", "-999.", "-999.000000", "-999") and float(r[c]) > -998 else np.nan
aod500 = np.array([num(r, "AOD_500nm") for r in sel]); alpha = np.array([num(r, "440-870_Angstrom_Exponent") for r in sel])
pw = np.array([num(r, "Precipitable_Water(cm)") for r in sel])
aod550 = aod500 * (550 / 500) ** (-alpha)
out["aeronet_n_obs_pm30min"] = len(sel)
out["aeronet_AOD550"] = round(float(np.nanmean(aod550)), 4) if len(sel) else None
out["aeronet_AOD550_range"] = [round(float(np.nanmin(aod550)), 4), round(float(np.nanmax(aod550)), 4)] if len(sel) else None
out["aeronet_PW_cm"] = round(float(np.nanmean(pw)), 4) if len(sel) else None
out["aeronet_angstrom"] = round(float(np.nanmean(alpha)), 3) if len(sel) else None
if len(sel):
    out["diff_AOT550_ours_minus_aeronet"] = round(out["ours_AOT550"][0] - out["aeronet_AOD550"], 4)
    out["diff_H2O_ours_minus_aeronet"] = round(out["ours_H2OSTR"][0] - out["aeronet_PW_cm"], 4)
print(json.dumps(out, indent=1))
