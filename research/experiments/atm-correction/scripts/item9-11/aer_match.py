"""Item 11: Tanager scenes on Drive whose footprint holds an AERONET site, and whether that site has data that day."""
import glob, json, urllib.request
from pathlib import Path
from shapely.geometry import shape, Point

sites = []
txt = urllib.request.urlopen("https://aeronet.gsfc.nasa.gov/aeronet_locations_v3.txt", timeout=60).read().decode()
for ln in txt.splitlines()[2:]:
    p = ln.split(",")
    try:
        sites.append((p[0], float(p[1]), float(p[2])))
    except (ValueError, IndexError):
        pass
print(len(sites), "AERONET sites")
D = "/content/drive/MyDrive/PS_11/tanager_datasets/scenes"
hits = []
for st in sorted(glob.glob(f"{D}/*/*_stac_item.json")):
    it = json.load(open(st))
    g = shape(it["geometry"]).buffer(0)
    for name, lon, lat in sites:
        if g.contains(Point(lon, lat)):
            hits.append((Path(st).parent.name, it["properties"]["datetime"], name, lat, lon))
print(len(glob.glob(f"{D}/*/*_stac_item.json")), "scenes,", len(hits), "with a site inside")
for sc, dt, name, lat, lon in hits:
    d = dt[:10]; y, m, dd = d.split("-")
    url = (f"https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site={name}&year={y}&month={int(m)}&day={int(dd)}"
           f"&year2={y}&month2={int(m)}&day2={int(dd)}&AOD15=1&AVG=10&if_no_html=1")
    try:
        body = urllib.request.urlopen(url, timeout=60).read().decode(errors="ignore")
        rows = [l for l in body.splitlines() if l.startswith(name)]
        times = [l.split(",")[2] for l in rows]
    except Exception as e:
        times = [f"ERR {e}"]
    print(sc, dt, name, f"{lat:.3f},{lon:.3f}", "L1.5 obs:", len(times), times[:3], "...", times[-3:] if times else "")
