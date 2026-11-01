"""Cut ang20151219t081738 L1 (rdn / loc_ort / obs_ort) and JPL corr to rows 2496-3096 (ang_prep.py's mid-strip
600-line window), as a native AVIRIS-NG L1 folder the entrypoint accepts. The georeference is rasterio's window
transform (rotation kept, unlike ang_prep.py's northing-only shift); spectral header keys are copied over."""
from pathlib import Path
import numpy as np, rasterio
from rasterio.windows import Window
from spectral.io import envi
SRC, R0, N = Path("/content/data/t11/ang"), 2496, 600
OUT = Path("/content/data/t11/ang600"); (OUT / "l1" / "ang20151219t081738").mkdir(parents=True, exist_ok=True)
(OUT / "l2").mkdir(exist_ok=True)
jobs = [(SRC / "ang20151219t081738" / f"ang20151219t081738_rdn_v2m2_{k}.bin", OUT / "l1" / "ang20151219t081738" / f"ang20151219t081738_rdn_v2m2_{k}.bin")
        for k in ("img", "loc_ort", "obs_ort")]
jobs.append((SRC / "l2" / "ang20151219t081738_corr_v2m2_img.bin", OUT / "l2" / "ang20151219t081738_corr_v2m2_img.bin"))
for src, dst in jobs:
    with rasterio.open(src) as d:
        w = Window(0, R0, d.width, N)
        prof = d.profile.copy()
        prof.update(height=N, transform=d.window_transform(w), driver="ENVI")
        prof["interleave"] = {"band": "bsq", "line": "bil", "pixel": "bip"}.get(d.interleaving.value.lower() if d.interleaving else "line", "bil")
        with rasterio.open(dst, "w", **prof) as o:
            o.write(d.read(window=w))
    old = envi.read_envi_header(str(src)[:-4] + ".hdr")
    new = envi.read_envi_header(str(dst)[:-4] + ".hdr")
    nb = int(old["bands"])
    for k, v in old.items():
        if k in ("samples", "lines", "bands", "header offset", "file type", "data type", "interleave", "byte order",
                 "map info", "coordinate system string", "description"):
            continue
        if isinstance(v, list) and len(v) > nb and k in ("fwhm", "wavelength", "bbl", "smoothing factors", "correction factors"):
            v = v[:nb]
        new[k] = v
    envi.write_envi_header(str(dst)[:-4] + ".hdr", new)
    aux = Path(str(dst) + ".aux.xml")
    if aux.exists():
        aux.unlink()
    print("cut", dst.name, "->", new["lines"], "lines; map info", new.get("map info", "")[:6])
(OUT / "l1" / "ang20151219t081738" / "LOC_ORT_DERIVED.txt").write_text("cut from the full line, rows 2496-3096\n")
print("CROP_DONE")
