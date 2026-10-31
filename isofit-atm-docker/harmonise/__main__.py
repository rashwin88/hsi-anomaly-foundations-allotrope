"""usage: python -m harmonise <model.npz> <atm_post.json> <obs> -> <rfl>_harm (ENVI BIL) + <rfl>_harm.json (provenance).
Covariates z = [AOT550, H2O, SZA/90, elev km (+ clim AOD550 for a 5-covariate model)]; SZA is the median of obs band 5."""
import hashlib, json, sys
import numpy as np
from spectral import envi
from . import apply, classes, model

NODATA = -9999.0
mpath, report, obs = sys.argv[1:4]
rep, (scene, parts) = json.load(open(report)), model.load(mpath)
if str(scene["sensor"]) != rep["sensor"]:
    sys.exit(f"model is for {scene['sensor']}, the run is {rep['sensor']}")
rfl, sza = rep["reflectance"], envi.open(obs + ".hdr").read_band(4)
z = [rep["scene_aot550"], rep["scene_h2o"], float(np.median(sza[sza > NODATA])) / 90, rep["scene_elev_km"]]
z += [rep["aot_prior"]["aod550_clim"]] if len(scene["zmu"]) == 5 else []   # KeyError/TypeError if the run had no prior
src = envi.open(rfl + ".hdr", rfl)
wl = np.array(src.metadata["wavelength"], float) * (1000 if float(src.metadata["wavelength"][-1]) < 100 else 1)
m, out = src.open_memmap(interleave="bil"), envi.create_image(rfl + "_harm.hdr", src.metadata, ext="", force=True).open_memmap(interleave="bil", writable=True)
for i in range(m.shape[0]):   # one line at a time: rows are pixels
    x = np.where(m[i].T > NODATA, m[i].T, np.nan)
    out[i] = np.nan_to_num(apply.harmonise(scene, parts, x, wl, z, classes.labels(x, wl)), nan=NODATA).T
json.dump({"model": mpath, "model_sha256": hashlib.sha256(open(mpath, "rb").read()).hexdigest(), "z": z,
           "gated": apply.gated(scene, apply.standardise(scene, z)[1]), "source": rfl,
           "skipped_parts": [n for n, p in parts.items() if apply.lacks_inputs(p, wl)]}, open(rfl + "_harm.json", "w"), indent=2)
