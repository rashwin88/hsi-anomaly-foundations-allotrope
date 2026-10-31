"""The quick-look report: one self-contained offline HTML page per scene, next to the product.

  python -m report --out <product dir> --work <work dir> [--ref <reference product>] [--ref-wavelength-file F]

Writes <name>_report.html (<name> = <fid>_L2_SR) with every image embedded (base64 PNG), so it opens on the
air-gapped VM without a network. It reads only what the run already wrote: the product GeoTIFF and its sidecars
(writers.py), scene.json and the L1 cube (readers.prep), ISOFIT's *_atm_interp, atm_post.json and stage_times.txt.
Sections:
  1  the run: sensor, rule, level, grade with the read / derived / assumed checklist, notes, stage times
  2  quick-looks: input radiance and output reflectance, true colour and NIR-red-green, outlier / snow overlay
  3  atmosphere: retrieved AOT550, water vapour and surface (pressure) elevation, with scene medians
  4  pixel statistics: median and 5-95 % spectra, sample pixels, histograms at four wavelengths, valid / flagged %
  5  optional comparison with a reference product (--ref: vendor L2, FLAASH, ...): matched pixels, spectra,
     per-band MAE / bias, scatter at three wavelengths
Images and statistics use a regular 1-in-N pixel grid (at most ~800 pixels a side), stated in each section.
A section that cannot be drawn says why; the report never fails the run (entrypoint.sh treats it as optional).

Reference products: an EnMAP L2A folder (SPECTRAL_IMAGE + METADATA.XML), a PRISMA L2 .he5, a Tanager surface
reflectance .h5, or any GeoTIFF / ENVI cube with wavelengths. Values above 5 (99th percentile) are read as x10000.
Pixels are paired by location (nearest reference pixel within one reference pixel spacing), so a map-projected
reference (PRISMA L2D, a reprojected FLAASH output) compares with a swath product; with no location on either side
and identical dimensions they are paired by index.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("report")
MAX_SIDE = 800
NODATA = -9999
HIST_NM = (550, 860, 1650, 2200)
SCATTER_NM = (660, 860, 1650)
FLAG_HIGH = 1


# --------------------------------------------------------------------------- small helpers

def band_at(wl: np.ndarray, nm: float, tol: float = 25.0) -> int | None:
    i = int(np.argmin(np.abs(wl - nm)))
    return i if abs(wl[i] - nm) <= tol else None


def png(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def stretch(rgb: np.ndarray, per_channel: bool) -> np.ndarray:
    """2-98 % stretch to 0..1; a common stretch keeps reflectance colours honest, per channel suits radiance."""
    out = np.zeros_like(rgb, np.float32)
    ok = np.all(np.isfinite(rgb), -1)
    if not ok.any():
        return out
    if per_channel:
        for k in range(3):
            lo, hi = np.percentile(rgb[..., k][ok], (2, 98))
            out[..., k] = (rgb[..., k] - lo) / max(hi - lo, 1e-9)
    else:
        lo, hi = 0.0, float(np.percentile(rgb[ok], 98))
        out = (rgb - lo) / max(hi - lo, 1e-9)
    out = np.clip(np.nan_to_num(out), 0, 1)
    out[~ok] = 1.0   # nodata shows white
    return out


def table(rows: list[list], head: list[str] | None = None) -> str:
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head) if head else ""
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{'<tr>' + h + '</tr>' if h else ''}{b}</table>"


def unavailable(why: str) -> str:
    return f'<p class="na">Not available: {html.escape(why)}</p>'


# --------------------------------------------------------------------------- product, L1 and atmosphere readers

class Product:
    """The written GeoTIFF on a 1-in-step grid: reflectance (rows, cols, bands) float, NaN for nodata."""

    def __init__(self, out_dir: Path):
        import rasterio
        from rasterio.enums import Resampling
        prov = sorted(out_dir.glob("*_L2_SR_provenance.json"))
        if not prov:
            raise FileNotFoundError(f"no *_L2_SR_provenance.json in {out_dir}")
        self.prov = json.loads(prov[0].read_text())
        self.name = prov[0].name[:-len("_provenance.json")]
        self.out = out_dir
        tif = out_dir / f"{self.name}.tif"
        with rasterio.open(tif) as d:
            self.lines, self.samples = d.height, d.width
            self.step = max(1, math.ceil(max(d.height, d.width) / MAX_SIDE))
            h, w = math.ceil(d.height / self.step), math.ceil(d.width / self.step)
            x = d.read(out_shape=(d.count, h, w), resampling=Resampling.nearest).astype(np.float32)
            self.wl = np.array([float(d.tags(i + 1)["wavelength"]) for i in range(d.count)])
            self.transform, self.crs = (d.transform, d.crs) if d.crs is not None and not d.transform.is_identity else (None, None)
        x[x == NODATA] = np.nan
        self.rfl = (x / float(self.prov["scale_factor"])).transpose(1, 2, 0)
        self.valid = np.all(np.isfinite(self.rfl), -1)
        self.masks = {}
        for tag in ("outlier", "snow"):
            p = out_dir / f"{self.name}_{tag}.tif"
            if p.exists():
                with rasterio.open(p) as d:
                    self.masks[tag] = d.read(1, out_shape=(h, w), resampling=Resampling.nearest)

    def lonlat(self) -> tuple[np.ndarray, np.ndarray] | None:
        """(lon, lat) of the sampled pixels: from the map transform, or the swath's _latlon.tif sidecar."""
        r = np.arange(0, self.lines, self.step) + 0.5
        c = np.arange(0, self.samples, self.step) + 0.5
        if self.transform is not None:   # rotation terms (AVIRIS-NG) included
            t = self.transform
            cc, rr = np.meshgrid(c, r)
            x, y = t.a * cc + t.b * rr + t.c, t.d * cc + t.e * rr + t.f
            if self.crs.to_epsg() == 4326:
                return x, y
            from pyproj import Transformer
            return Transformer.from_crs(self.crs, 4326, always_xy=True).transform(x, y)
        p = self.out / f"{self.name}_latlon.tif"
        if p.exists():
            import rasterio
            with rasterio.open(p) as d:
                lat, lon = d.read()[:, ::self.step, ::self.step]
            return lon.astype(np.float64), lat.astype(np.float64)
        return None


def read_l1(work: Path, fid: str, step: int, nms: tuple) -> tuple[np.ndarray, str] | None:
    """Three bands of the cube ISOFIT was given (radiance, or the input reflectance on pass-through)."""
    from spectral.io import envi
    for stem, what in ((f"{fid}_rdn", "input radiance (L1)"), (f"{fid}_rfl", "input (surface reflectance)")):
        p = work / stem
        if Path(str(p) + ".hdr").exists():
            img = envi.open(str(p) + ".hdr", str(p))
            wl = np.asarray(img.metadata["wavelength"], float)
            wl = wl * 1000 if wl.max() < 100 else wl
            idx = [band_at(wl, nm) for nm in nms]
            if None in idx:
                return None
            m = img.open_memmap(interleave="bip")
            x = np.asarray(m[::step, ::step, idx], np.float32)
            x[~np.isfinite(x) | (x <= -9990)] = np.nan
            return x, what
    return None


def read_atm(work: Path, step: int) -> dict[str, np.ndarray]:
    import rasterio
    found = [p for p in (work / "isofit" / "output").glob("*_atm_interp") if "_subs_" not in p.name]
    if not found:
        return {}
    out = {}
    with rasterio.open(found[0]) as d:
        names = [(n or "").upper() for n in d.descriptions]
        for key in ("AOT550", "H2OSTR", "SURFACE_ELEVATION_KM"):
            if key in names:
                b = d.read(names.index(key) + 1)[::step, ::step].astype(np.float32)
                b[b <= -9990] = np.nan
                out[key] = b
    return out


# --------------------------------------------------------------------------- reference products

def open_reference(path: str, wl_file: str | None = None):
    """(wl ascending, (lines, samples), read_rows(r0, r1) -> (rows, bands, samples) float NaN-invalid,
    lonlat(r0, r1) -> (lon, lat) or None, description)."""
    from readers import cube
    p = Path(path)
    if p.is_dir():
        imgs = [q for q in sorted(p.glob("*SPECTRAL_IMAGE*")) if q.suffix.upper() in (".TIF", ".TIFF")]
        if not imgs:
            raise FileNotFoundError(f"{p}: no *SPECTRAL_IMAGE*.TIF in the folder")
        p = imgs[0]
    n = p.name.lower()
    if n.endswith(".he5") and n.startswith("prs_l2"):
        return _open_prisma_l2(p)
    if n.endswith((".h5", ".he5")):
        from readers.tanager import open_tanager
        info, extra = open_tanager(str(p))
        loc = extra.get("location")
        ll = (lambda r0, r1, f=loc.value: f(r0, r1)) if loc is not None and loc.value is not None else None
    else:
        info = cube.open_geotiff(str(p), wl_file) if n.endswith((".tif", ".tiff")) else \
            cube.open_envi(cube.find_hdr(str(p)) or str(p), wl_file)
        f = cube.lonlat_reader(info.transform, info.crs)
        ll = (lambda r0, r1, f=f, ns=info.shape[2]: f(r0, r1, ns)) if f else None
    info = cube.sort_bands(info)
    return info.wl, (info.shape[0], info.shape[2]), info.read_rows, ll, f"{p.name} ({info.fmt})"


def _open_prisma_l2(p: Path):
    import h5py
    f = h5py.File(p, "r")
    a = f.attrs
    lvl = next(k for k in f["HDFEOS/SWATHS"].keys() if "Data Fields/VNIR_Cube" in f[f"HDFEOS/SWATHS/{k}"])
    sw = f[f"HDFEOS/SWATHS/{lvl}"]
    fams = []
    for fam in ("Vnir", "Swir"):
        wl = np.asarray(a[f"List_Cw_{fam}"], float)
        keep = np.asarray(a[f"List_Cw_{fam}_Flags"]).astype(bool) & (wl > 0)
        lo, hi = float(a[f"L2Scale{fam}Min"]), float(a[f"L2Scale{fam}Max"])
        fams.append((wl[keep], keep, lo, hi, sw[f"Data Fields/{fam.upper()}_Cube"]))
    wl = np.concatenate([x[0] for x in fams])
    order = np.argsort(wl)
    L, _, S = fams[0][4].shape

    def read_rows(r0, r1):
        parts = []
        for _, keep, lo, hi, ds in fams:
            dn = ds[r0:r1][:, keep, :]
            parts.append(np.where(dn == 0, np.nan, lo + dn.astype(np.float32) * (hi - lo) / 65535))
        return np.concatenate(parts, 1)[:, order, :].astype(np.float32)

    lat, lon = sw["Geolocation Fields/Latitude"], sw["Geolocation Fields/Longitude"]
    return wl[order], (L, S), read_rows, lambda r0, r1: (np.asarray(lon[r0:r1], float), np.asarray(lat[r0:r1], float)), \
        f"{p.name} (PRISMA {lvl})"


def match_reference(prod: Product, ref) -> dict:
    """Pair sampled product pixels with reference pixels; return both spectra on the product's bands."""
    from scipy.spatial import cKDTree
    wl_r, (Lr, Sr), read_rows, ll, desc = ref
    h, w = prod.valid.shape
    ours_ll = prod.lonlat()
    if ours_ll is not None and ll is not None:
        lon_p, lat_p = ours_ll
        lat0 = float(np.nanmedian(lat_p))
        k = 111320.0
        xy = lambda lo, la: np.stack([lo * k * math.cos(math.radians(lat0)), la * k], -1)
        lon_r, lat_r = ll(0, Lr)
        pts_r = xy(np.asarray(lon_r, float).ravel(), np.asarray(lat_r, float).ravel())
        okr = np.all(np.isfinite(pts_r), 1)
        tree = cKDTree(pts_r[okr])
        spacing = float(np.median(tree.query(pts_r[okr][:: max(1, okr.sum() // 5000)], k=2)[0][:, 1]))
        dist, j = tree.query(xy(lon_p, lat_p).reshape(-1, 2))
        flat_r = np.flatnonzero(okr)[np.where(np.isfinite(dist), j, 0)]
        hit = np.isfinite(dist) & (dist <= spacing)
        how = f"by location: nearest reference pixel within one reference pixel spacing ({spacing:.1f} m)"
    elif (Lr, Sr) == (prod.lines, prod.samples):
        rr, cc = np.meshgrid(np.arange(0, prod.lines, prod.step), np.arange(0, prod.samples, prod.step), indexing="ij")
        flat_r, hit, how = (rr * Sr + cc).ravel(), np.ones(rr.size, bool), "by pixel index (same dimensions, no location)"
    else:
        raise ValueError(f"no location on one side and dimensions differ ({Lr}x{Sr} vs {prod.lines}x{prod.samples})")
    hit &= prod.valid.ravel()
    if "outlier" in prod.masks:
        hit &= prod.masks["outlier"].ravel() != FLAG_HIGH
    want = flat_r[hit]
    rw = want // Sr
    spec = np.full((want.size, wl_r.size), np.nan, np.float32)
    # only the reference rows that hold a paired pixel, read in runs of consecutive rows (a long airborne flight
    # line is tens of GB; the 1-in-N product grid needs a small fraction of its rows)
    rows = np.unique(rw)
    runs = np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1) if rows.size else []
    for run in runs:
        for r0 in range(int(run[0]), int(run[-1]) + 1, 128):
            r1 = min(r0 + 128, int(run[-1]) + 1)
            sel = (rw >= r0) & (rw < r1)
            if sel.any():
                spec[sel] = read_rows(r0, r1)[rw[sel] - r0, :, want[sel] % Sr]
    spec[spec <= -9990] = np.nan
    # reference bands that are empty almost everywhere (its own water bands) do not invalidate a pixel
    used = np.mean(np.isfinite(spec), 0) > 0.5 if want.size else np.zeros(wl_r.size, bool)
    spec, wl_r = spec[:, used], wl_r[used]
    ok = np.all(np.isfinite(spec), 1) & np.any(spec != 0, 1)
    scale = 1e-4 if ok.any() and np.percentile(spec[ok], 99) > 5 else 1.0
    # reference onto the product bands: linear in wavelength, only inside its range and near one of its bands
    near = np.array([wl_r.size > 1 and np.min(np.abs(wl_r - x)) <= 15 and wl_r[0] <= x <= wl_r[-1] for x in prod.wl])
    ref_on = np.full((ok.sum(), prod.wl.size), np.nan, np.float32)
    good = spec[ok] * scale
    for b in np.flatnonzero(near):
        i = np.searchsorted(wl_r, prod.wl[b]).clip(1, wl_r.size - 1)
        t = (prod.wl[b] - wl_r[i - 1]) / (wl_r[i] - wl_r[i - 1])
        ref_on[:, b] = (1 - t) * good[:, i - 1] + t * good[:, i]
    ours = prod.rfl.reshape(-1, prod.wl.size)[hit][ok]
    return {"desc": desc, "how": how, "scale": scale, "n": int(ok.sum()), "n_sampled": int(prod.valid.sum()),
            "ours": ours, "ref": ref_on, "bands": near}


# --------------------------------------------------------------------------- sections

def section_run(prod: Product, scene: dict, work: Path) -> str:
    p = prod.prov
    rows = [["Scene", scene["fid"]], ["Input", scene["input"]], ["Format", scene["format"]],
            ["Sensor rule", f"{scene['rule']} (ISOFIT sensor {scene['isofit_sensor']})"],
            ["Input level", scene["level"] + (" (passed through, no correction)" if scene.get("pass_through") else "")],
            ["Date", scene["date"]], ["Centre", f"{scene['centre_lat']}, {scene['centre_lon']}" if scene["centre_lat"] is not None else "unknown"],
            ["Pixel", f"{scene.get('pixel_m')} m"], ["Size", f"{scene['lines']} lines x {scene['samples']} samples"],
            ["Bands", f"{p['bands_kept']} written, {len(p['bands_dropped_nm'])} dropped (water / outside windows / empty)"],
            ["Georeference", p["georeference"]], ["Scale", f"int16 = reflectance x {p['scale_factor']:g}, nodata {p['nodata']}"]]
    harm = p.get("harmonisation")
    rows.append(["Harmonisation", harm.get("model", "applied") if isinstance(harm, dict) else "not applied"])
    g = scene["grade"]
    out = (f'<p class="grade g{g}">Grade {g}</p><p class="small">A: everything from metadata; B: some derived; '
           "C: view, altitude, noise model or DEM assumed; D: time, location or radiance scale assumed (best effort).</p>")
    out += table(rows)
    out += "<h3>What was read, derived and assumed</h3>" + table(
        [[r["item"], r["source"], r["value"], r["note"]] for r in scene["checklist"]], ["item", "source", "value", "note"])
    if scene.get("notes"):
        out += "<h3>Notes</h3><ul>" + "".join(f"<li>{html.escape(n)}</li>" for n in scene["notes"]) + "</ul>"
    st = work / "stage_times.txt"
    if st.exists():
        marks = [ln.split() for ln in st.read_text().splitlines() if len(ln.split()) == 2]
        t = [[a, f"{(int(b2) - int(b1)) / 60:.1f} min"] for (a, b1), (_, b2) in zip(marks, marks[1:])]
        if marks:
            t.append(["total so far", f"{(int(marks[-1][1]) - int(marks[0][1])) / 60:.1f} min"])
        out += "<h3>Stage times</h3>" + table(t, ["stage", "time"])
    return out


def section_looks(prod: Product, work: Path, fid: str) -> str:
    import matplotlib.pyplot as plt
    tc, fc = (640, 550, 470), (860, 650, 550)
    out = f'<p class="small">1-in-{prod.step} pixel grid. White: nodata.</p>'
    idx_tc = [band_at(prod.wl, n) for n in tc]
    idx_fc = [band_at(prod.wl, n) for n in fc]
    l1 = read_l1(work, fid, prod.step, tc + fc)
    n = 2 + (2 if l1 else 0)
    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    k = 0
    if l1:
        x, what = l1
        for sl, lab in ((slice(0, 3), "true colour"), (slice(3, 6), "NIR-red-green")):
            ax[k].imshow(stretch(x[..., sl], "radiance" in what)); ax[k].set_title(f"{what}\n{lab}", fontsize=9); k += 1
    for idx, lab in ((idx_tc, "true colour"), (idx_fc, "NIR-red-green")):
        if None in idx:
            ax[k].text(0.5, 0.5, "bands missing", ha="center"); k += 1; continue
        ax[k].imshow(stretch(prod.rfl[..., idx], False)); ax[k].set_title(f"L2 surface reflectance\n{lab}", fontsize=9); k += 1
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    out += png(fig)
    if not l1:
        out += unavailable("input-side views: no <fid>_rdn / _rfl cube with the needed bands in the work dir")
    if prod.masks and None not in idx_tc:
        base = stretch(prod.rfl[..., idx_tc], False) * 0.6 + 0.2
        lines = []
        if "snow" in prod.masks:
            base[prod.masks["snow"] == 1] = (0.2, 0.6, 1.0); lines.append("blue: snow (NDSI)")
        if "outlier" in prod.masks:
            base[prod.masks["outlier"] == FLAG_HIGH] = (1.0, 0.0, 1.0); lines.append("magenta: reflectance above 2.5, kept and flagged")
            base[prod.masks["outlier"] == 2] = (1.0, 0.5, 0.0); lines.append("orange: below -0.2, set to nodata")
        fig, a = plt.subplots(figsize=(6, 6))
        a.imshow(base); a.set_xticks([]); a.set_yticks([]); a.set_title("flags over the true-colour image", fontsize=9)
        out += "<h3>Flags</h3>" + png(fig) + "<p class='small'>" + "; ".join(lines) + "</p>"
    return out


def section_atm(prod: Product, work: Path) -> str:
    import matplotlib.pyplot as plt
    atm = read_atm(work, prod.step)
    rep = prod.prov.get("atm_post") or {}
    if not atm:
        return unavailable("no ISOFIT *_atm_interp in the work dir (input passed through, or work dir missing)")
    lab = {"AOT550": "AOT at 550 nm", "H2OSTR": "water vapour (g/cm2)", "SURFACE_ELEVATION_KM": "surface pressure elevation (km)"}
    fig, ax = plt.subplots(1, len(atm), figsize=(4.6 * len(atm), 4.4))
    ax = np.atleast_1d(ax)
    rows = []
    for a, (k, v) in zip(ax, atm.items()):
        lo, hi = np.nanpercentile(v, (2, 98)) if np.isfinite(v).any() else (0, 1)
        im = a.imshow(v, vmin=lo, vmax=hi, cmap="viridis"); a.set_xticks([]); a.set_yticks([])
        med = float(np.nanmedian(v))
        a.set_title(f"{lab[k]}\nmedian {med:.3f}", fontsize=9)
        fig.colorbar(im, ax=a, fraction=0.046)
        rows.append([lab[k], f"{med:.3f}", f"{np.nanpercentile(v, 5):.3f} .. {np.nanpercentile(v, 95):.3f}"])
    out = png(fig) + table(rows, ["retrieved", "median", "5-95 %"])
    if rep:
        pr = rep.get("aot_prior") or {}
        extra = [["AOT prior", f"climatology {pr.get('aod550_clim')}" if pr else "none (free AOT)"],
                 ["Adjacency correction", "applied" if rep.get("adjacency_applied") else "skipped"],
                 ["Terrain correction", "applied" if rep.get("terrain_applied") else "skipped"],
                 ["Steep terrain fraction", f"{100 * rep.get('steep_fraction', 0):.1f} %"]]
        out += table(extra)
    return out


def section_stats(prod: Product) -> str:
    import matplotlib.pyplot as plt
    x, ok, wl = prod.rfl, prod.valid, prod.wl
    if not ok.any():
        return unavailable("no valid pixel in the product")
    v = x[ok]
    out = f'<p class="small">{ok.sum():,} valid pixels on the 1-in-{prod.step} grid.</p>'
    gap = lambda y: np.where(np.r_[False, np.diff(wl) > 30], np.nan, y)   # break the line across dropped bands
    p5, p50, p95 = np.percentile(v, (5, 50, 95), axis=0)
    fig, a = plt.subplots(figsize=(9, 3.8))
    a.fill_between(wl, gap(p5), gap(p95), alpha=0.3, label="5-95 %"); a.plot(wl, gap(p50), label="median")
    a.set_xlabel("wavelength (nm)"); a.set_ylabel("reflectance"); a.legend(); a.grid(alpha=0.3)
    a.set_title("scene reflectance distribution", fontsize=10)
    out += png(fig)

    picks = {}
    mean = v.mean(1)
    picks["brightest"] = int(np.argmax(mean))
    picks["darkest"] = int(np.argmin(mean))
    ir, ig, inir = band_at(wl, 660), band_at(wl, 560), band_at(wl, 860)
    if None not in (ir, inir):
        ndvi = (v[:, inir] - v[:, ir]) / (v[:, inir] + v[:, ir] + 1e-6)
        picks["greenest (max NDVI)"] = int(np.argmax(ndvi))
    if None not in (ig, inir):
        ndwi = (v[:, ig] - v[:, inir]) / (v[:, ig] + v[:, inir] + 1e-6)
        if (ndwi > 0.2).any():
            picks["water (NDWI > 0.2, darkest NIR)"] = int(np.flatnonzero(ndwi > 0.2)[np.argmin(v[ndwi > 0.2, inir])])
    if "outlier" in prod.masks:
        fl = np.flatnonzero(prod.masks["outlier"][ok] == FLAG_HIGH)[:2]
        for i, j in enumerate(fl):
            picks[f"flagged bright {i + 1}"] = int(j)
    fig, a = plt.subplots(figsize=(9, 3.8))
    for k, j in picks.items():
        a.plot(wl, gap(v[j]), label=k)
    a.set_xlabel("wavelength (nm)"); a.set_ylabel("reflectance"); a.legend(fontsize=8); a.grid(alpha=0.3)
    a.set_title("sample pixels", fontsize=10)
    out += "<h3>Sample pixels</h3>" + png(fig)

    hb = [(nm, band_at(wl, nm)) for nm in HIST_NM]
    hb = [(nm, b) for nm, b in hb if b is not None]
    fig, ax = plt.subplots(1, len(hb), figsize=(3.4 * len(hb), 2.8))
    for a, (nm, b) in zip(np.atleast_1d(ax), hb):
        a.hist(v[:, b], bins=80, range=(min(-0.05, v[:, b].min()), max(0.8, np.percentile(v[:, b], 99.5))))
        a.set_title(f"{wl[b]:.0f} nm", fontsize=9); a.set_yticks([])
    out += "<h3>Histograms</h3>" + png(fig)

    rep = prod.prov.get("atm_post") or {}
    n_all = prod.valid.size
    rows = [["valid (1-in-N grid)", f"{100 * ok.sum() / n_all:.2f} %"],
            ["negative anywhere (valid pixels)", f"{100 * np.mean(np.any(v < 0, 1)):.2f} %"],
            ["above 1.0 anywhere (valid pixels)", f"{100 * np.mean(np.any(v > 1, 1)):.2f} %"]]
    if rep:
        L, S = prod.lines, prod.samples
        rows += [["below -0.2, set to nodata (all pixels)", f"{rep.get('outlier_pixels_nodata', 0):,} ({100 * rep.get('outlier_pixels_nodata', 0) / (L * S):.3f} %)"],
                 ["above 2.5, kept and flagged (all pixels)", f"{rep.get('bright_pixels_flagged', 0):,} ({100 * rep.get('bright_pixels_flagged', 0) / (L * S):.3f} %)"],
                 ["snow (NDSI)", f"{100 * rep.get('snow_fraction', 0):.2f} %"]]
    out += "<h3>Pixel counts</h3>" + table(rows)
    return out


def section_ref(prod: Product, ref_path: str, wl_file: str | None) -> str:
    import matplotlib.pyplot as plt
    m = match_reference(prod, open_reference(ref_path, wl_file))
    if m["n"] < 10:
        return unavailable(f"only {m['n']} product pixels pair with a valid reference pixel ({m['how']})")
    o, r, b = m["ours"], m["ref"], m["bands"]
    d = (o - r)[:, b]
    wl = prod.wl[b]
    mae, bias, rmse = np.nanmean(np.abs(d), 0), np.nanmean(d, 0), np.sqrt(np.nanmean(d ** 2, 0))
    out = table([["Reference", m["desc"]], ["Pairing", m["how"]], ["Reference scale", f"x{m['scale']:g}"],
                 ["Pixels compared", f"{m['n']:,} of {m['n_sampled']:,} valid sampled product pixels (flagged bright excluded)"],
                 ["Bands compared", f"{int(b.sum())} of {prod.wl.size}"],
                 ["MAE (all bands)", f"{np.nanmean(np.abs(d)):.4f}"], ["Bias, ours - reference", f"{np.nanmean(d):+.4f}"],
                 ["RMSE (all bands)", f"{np.sqrt(np.nanmean(d ** 2)):.4f}"]])
    gap = lambda w, y: np.where(np.r_[False, np.diff(w) > 30], np.nan, y)
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
    ax[0].plot(wl, gap(wl, np.nanmedian(o[:, b], 0)), label="ours (median)")
    ax[0].plot(wl, gap(wl, np.nanmedian(r[:, b], 0)), label="reference (median)")
    ax[0].set_xlabel("wavelength (nm)"); ax[0].set_ylabel("reflectance"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(wl, gap(wl, mae), label="MAE"); ax[1].plot(wl, gap(wl, bias), label="bias (ours - ref)")
    ax[1].plot(wl, gap(wl, rmse), label="RMSE", alpha=0.6); ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("wavelength (nm)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    out += png(fig)
    sb = [(nm, band_at(prod.wl, nm)) for nm in SCATTER_NM]
    sb = [(nm, i) for nm, i in sb if i is not None and b[i]]
    if sb:
        fig, ax = plt.subplots(1, len(sb), figsize=(4 * len(sb), 3.8))
        for a, (nm, i) in zip(np.atleast_1d(ax), sb):
            ok = np.isfinite(o[:, i]) & np.isfinite(r[:, i])
            hi = float(np.percentile(np.r_[o[ok, i], r[ok, i]], 99.5))
            a.hexbin(r[ok, i], o[ok, i], gridsize=60, bins="log", extent=(0, hi, 0, hi), mincnt=1)
            a.plot([0, hi], [0, hi], "k", lw=0.7)
            a.set_xlabel("reference"); a.set_ylabel("ours"); a.set_title(f"{prod.wl[i]:.0f} nm", fontsize=9)
        out += png(fig)
    rows = [[f"{w:.1f}", f"{x:.4f}", f"{y:+.4f}", f"{z:.4f}"] for w, x, y, z in zip(wl, mae, bias, rmse)]
    out += "<details><summary>per-band table</summary>" + table(rows, ["nm", "MAE", "bias", "RMSE"]) + "</details>"
    return out


CSS = """body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1400px;
padding:16px;color:#1d1d1f;background:#fff}h1{font-size:22px}h2{font-size:18px;border-bottom:1px solid #ddd;
padding-bottom:4px;margin-top:32px}h3{font-size:15px}table{border-collapse:collapse;font-size:13px;margin:8px 0}
td,th{border:1px solid #ddd;padding:3px 8px;text-align:left;vertical-align:top}th{background:#f4f4f6}
img{max-width:100%;height:auto}.small{font-size:12px;color:#555}.na{color:#a15c00}
.grade{display:inline-block;font-size:20px;font-weight:600;padding:4px 14px;border-radius:6px;color:#fff}
.gA{background:#1b7f3b}.gB{background:#4a8f2a}.gC{background:#b07d00}.gD{background:#b3261e}
details{margin:8px 0}"""


def build(out_dir: Path, work: Path, ref: str | None = None, ref_wl: str | None = None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    t0 = time.perf_counter()
    prod = Product(out_dir)
    scene = prod.prov["scene"]
    sections = [("1. The run", lambda: section_run(prod, scene, work)),
                ("2. Quick-looks", lambda: section_looks(prod, work, scene["fid"])),
                ("3. Retrieved atmosphere", lambda: section_atm(prod, work)),
                ("4. Pixel statistics", lambda: section_stats(prod))]
    if ref:
        sections.append(("5. Comparison with the reference", lambda: section_ref(prod, ref, ref_wl)))
    body = []
    for title, f in sections:
        try:
            body.append(f"<h2>{title}</h2>" + f())
        except Exception as e:   # noqa: BLE001 - one section failing must not lose the others
            log.warning("section %r failed: %s", title, e)
            body.append(f"<h2>{title}</h2>" + unavailable(f"{type(e).__name__}: {e}"))
    page = (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(prod.name)}</title><style>{CSS}</style>"
            f"</head><body><h1>{html.escape(prod.name)}: L1 to L2 surface reflectance</h1>"
            f"<p class='small'>ISOFIT 4.1.5 + sRTMnet. Report written {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}."
            f"</p>{''.join(body)}</body></html>")
    p = out_dir / f"{prod.name}_report.html"
    p.write_text(page, encoding="utf-8")
    log.info("wrote %s (%.1f MB) in %.1f s", p, p.stat().st_size / 2**20, time.perf_counter() - t0)
    return p


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--out", required=True, type=Path, help="product dir (writers.py output)")
    a.add_argument("--work", required=True, type=Path, help="the run's work dir (scene.json, L1 cube, isofit/)")
    a.add_argument("--ref", help="reference product to compare with")
    a.add_argument("--ref-wavelength-file", help="band centres for a reference that carries none")
    o = a.parse_args(argv)
    build(o.out, o.work, o.ref, o.ref_wavelength_file)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
