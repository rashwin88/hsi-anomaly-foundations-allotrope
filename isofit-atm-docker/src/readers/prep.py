"""Any supported delivery -> ISOFIT's input cubes plus scene.json (what was known, derived, assumed; the grade).

  python -m readers.prep --input <file or folder> --workdir <dir> [--dem <vrt>] [--sensor S]
                         [--input-level radiance|toa|sr] [--radiance-scale X]

Native products keep their validated preprocessors (EnMAP L1C folder, PRISMA L1 HE5) or are linked as delivered
(AVIRIS-NG L1 folder: ISOFIT reads its rdn / loc_ort / obs_ort as they are). Everything else - any sensor as ENVI or
GeoTIFF, and Tanager HDF5 - goes through the generic path: cube.py / tanager.py read it, the fallback ladder fills
time, location, sun and view, level.py decides what the numbers are, and this module writes:
  <fid>_rdn   radiance, uW/cm^2/sr/nm, float32 BIL, -9999 for invalid pixels (TOA reflectance converted with
              ISOFIT's own solar irradiance; bands dead in more than 5% of pixels dropped and listed)
  <fid>_loc   lon, lat, elevation (m), float32 BIP
  <fid>_obs   ISOFIT's 10 observation bands, float32 BIP
  <fid>_rfl   only for a surface-reflectance delivery that passes level.py's SR rule: the cube to pass through
  wavelengths.txt, season.txt, fid.txt, scene.json
Grades and flags never stop a run. It exits non-zero with a message only when the physics cannot run, and every such
stop has an operator override, recorded in the checklist as "set by the operator":
  raw counts with no calibration           --radiance-scale <factor to uW/cm^2/sr/nm>
  level undecidable / values fit no scale  --input-level radiance|toa|sr
  no spectral axis in the delivery         --wavelength-file <wl [fwhm] table>
  surface reflectance passed through       --input-level toa (or radiance) corrects it anyway
A native EnMAP / PRISMA product without a DEM runs on a flat 0 m DEM, flagged (dem: assumed).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import cube, geometry, level, sensors
from .scene import Fact, Scene

log = logging.getLogger("readers.prep")
NODATA = -9999.0
ROWS = 64
ASSUMED_DATE = datetime(2026, 4, 4, 6, 0, tzinfo=timezone.utc)   # Earth-Sun distance 1.000 AU (+/-0.1%)


def detect(p: Path) -> tuple[str, Path]:
    """(format, the file or folder to read)."""
    if p.is_dir():
        if glob.glob(str(p / "*-METADATA.XML")) and glob.glob(str(p / "*SPECTRAL_IMAGE*")):
            return "enmap_l1c", p
        rdn = glob.glob(str(p / "ang*_rdn_*_img.hdr"))
        if rdn and glob.glob(str(p / "*_obs_ort.hdr")) and glob.glob(str(p / "*_loc_ort.hdr")):
            return "aviris_ng_l1", p
        cands = sorted(glob.glob(str(p / "*.he5")) + glob.glob(str(p / "*.h5")) + glob.glob(str(p / "*.hdr"))
                       + glob.glob(str(p / "*.tif")) + glob.glob(str(p / "*.tiff")))
        cands = [c for c in cands if "radiance" in c or len(cands) == 1] or cands
        if len(cands) != 1:
            raise SystemExit(f"{p}: cannot tell which of {len(cands)} files is the cube; pass it directly")
        return detect(Path(cands[0]))
    n = p.name.lower()
    if "spectral_image" in n and any("L1C" in x.name for x in p.parent.glob("*-METADATA.XML")):
        return "enmap_l1c", p.parent   # the cube alone is raw counts; its folder carries the gains
    if n.endswith(".he5") and n.startswith("prs_l1"):
        return "prisma_l1", p
    if n.endswith((".h5", ".he5")):
        return "tanager_h5", p
    if n.endswith((".tif", ".tiff")):
        return "geotiff", p
    hdr = cube.find_hdr(str(p))
    if hdr:
        return "envi", Path(hdr)
    raise SystemExit(f"{p}: not a recognised product, ENVI cube or GeoTIFF")


# --------------------------------------------------------------------------- generic path

def build(fmt: str, path: Path, sensor: str | None, wl_file: str | None = None) -> Scene:
    extra = {}
    if fmt == "tanager_h5":
        from .tanager import open_tanager
        info, extra = open_tanager(str(path))
    else:
        info = cube.open_envi(str(path), wl_file) if fmt == "envi" else cube.open_geotiff(str(path), wl_file)
    info = cube.sort_bands(info)
    text = " ".join(f"{k}={v}" for k, v in info.meta.items() if isinstance(v, str))
    sens = Fact(sensor, "metadata", "set by the operator") if sensor else sensors.identify(str(path), text, info.wl, info.fwhm)
    fwhm = (Fact(info.fwhm, "metadata", "delivered") if info.fwhm is not None and np.all(info.fwhm > 0)
            else Fact(cube.derive_fwhm(info.wl), "derived", "from band spacing (x1.0): no FWHM delivered"))
    t = extra.get("time") or sensors.acquisition_time(str(path), info.meta)
    loc = extra.get("location")
    if loc is None:
        ll = cube.lonlat_reader(info.transform, info.crs)
        loc = (Fact(lambda r0, r1, ns=info.shape[2]: ll(r0, r1, ns), "metadata", f"georeference {info.crs}")
               if ll else Fact(None, "missing", "no CRS / map info: no DEM, lat/lon 0 placeholders"))
    sun = extra.get("sun")
    if sun is None:
        if t.value is not None and loc.value is not None:
            sun = Fact(lambda r0, r1, f=loc.value, tt=t.value: geometry.sun_position(tt, *f(r0, r1)[::-1]),
                       "derived", "NOAA solar position from the acquisition time and pixel location")
        else:
            sun = Fact(lambda r0, r1, ns=info.shape[2]: (np.full((r1 - r0, ns), geometry.ASSUMED_SZA),
                                                         np.full((r1 - r0, ns), geometry.ASSUMED_SAA)),
                       "assumed", f"sun zenith {geometry.ASSUMED_SZA} deg, azimuth {geometry.ASSUMED_SAA} deg: "
                                  "time or location unknown")
    view = extra.get("view") or Fact(lambda r0, r1, ns=info.shape[2]: (np.zeros((r1 - r0, ns)), np.zeros((r1 - r0, ns))),
                                     "assumed", "nadir: no view angles delivered")
    sid = sens.value if sens.value in sensors.SENSORS else "unknown"
    alt = sensors.SENSORS[sid]["alt_km"]
    altitude = (Fact(None, "metadata", "from the delivered path length") if extra.get("path_length")
                else Fact(alt, "assumed", f"nominal {sid} orbit") if alt else
                Fact(5.0, "assumed", "airborne with no obs: 5 km above sea level"))
    stem = path.stem if path.suffix.lower() != ".hdr" else Path(str(path)[:-4]).name
    sc = Scene(str(path), fmt, sens, info.shape, info.interleave, info.wl, fwhm, info.read_rows,
               Fact(None, "missing"), Fact(None, "missing"), t, loc, sun, view, altitude,
               path_length=extra.get("path_length"), notes=list(info.notes))
    sc.isofit_sensor = "tanager" if fmt == "tanager_h5" else f"NA-{(t.value or ASSUMED_DATE):%Y%m%d}"
    sc.fid = stem.replace(".", "_") if fmt != "tanager_h5" else stem[:23]
    sc._extra, sc._dtype, sc._info = extra, info.dtype, info
    if t.value is None:
        sc.time = Fact(ASSUMED_DATE, "assumed", "no acquisition time: Earth-Sun distance 1 AU (date 2026-04-04)")
    return sc


def decide_level(sc: Scene, forced: str | None, radiance_scale: float | None) -> np.ndarray:
    """Sets sc.level, sc.units, sc.radiance_scale. Returns the pixel sample it used."""
    smp = level.sample_pixels(sc.read_rows, sc.lines)
    integer = np.dtype(sc._dtype.lstrip("<>")).kind in "iu" and "_scale" not in sc._info.meta
    res = level.classify(smp, sc.wl, integer=integer)
    lv = {"radiance": "radiance", "toa": "toa_reflectance", "sr": "surface_reflectance"}.get(forced or "", None)
    native = sc._extra.get("level")
    if lv:
        sc.level = Fact(lv, "metadata", f"set by the operator (tests said {res['level']}: {res.get('reason')})")
    elif native == "surface_reflectance":   # the product's own variable name, not a guess from the numbers
        sc.level = Fact(native, "metadata", f"product variable surface_reflectance (tests: {res['level']})")
    elif native == "radiance" and res["level"] in ("radiance", "unknown"):
        sc.level = Fact("radiance", "metadata", "product type (toa_radiance); tests agree" if res["level"] == "radiance"
                        else "product type (toa_radiance); tests inconclusive")
    else:
        sc.level = Fact(res["level"], "derived", res.get("reason", ""))
    sc.level_tests = res
    if sc.level.value == "dn":
        if radiance_scale is None:
            raise SystemExit("input looks like raw counts with no calibration (integer, sun-shaped spectra, no "
                             "gain/offset). Pass --radiance-scale <factor to uW/cm^2/sr/nm> or the calibrated product.")
        sc.level = Fact("radiance", "metadata", f"raw counts x {radiance_scale} (operator)")
        sc.radiance_scale = Fact(radiance_scale, "metadata", "set by the operator")
        sc.units = Fact("counts", "metadata", "operator scale applied")
    elif sc.level.value == "unknown":
        raise SystemExit(f"cannot tell radiance from reflectance ({res.get('reason')}); pass --input-level")
    elif sc.level.value == "radiance" and sc.radiance_scale.source == "metadata" and sc.radiance_scale.value:
        pass   # a native preprocessor already wrote uW/cm^2/sr/nm
    elif sc.level.value == "radiance":
        if radiance_scale is not None:
            sc.radiance_scale = Fact(radiance_scale, "metadata", "set by the operator")
            sc.units = Fact("operator", "metadata", "--radiance-scale")
        else:
            keys = {k: v for k, v in sc._info.meta.items() if isinstance(v, str) and
                    any(w in k for w in ("unit", "radiance", "description"))}
            sza = float(np.nanmean(sc.sun.value(sc.lines // 2, sc.lines // 2 + 1)[0]))
            d2 = geometry.earth_sun_au(sc.time.value) ** 2
            e0 = []   # irradiance only when the unit has to be inferred (a labelled unit returns first)
            def per_unit(i):
                if not e0:
                    e0.append(geometry.solar_irradiance(sc.wl, sc.fwhm.value))
                return np.pi * d2 / (e0[0][i] * max(np.cos(np.radians(sza)), 0.1))
            sc.units, sc.radiance_scale = sensors.radiance_units(keys, smp, sc.wl, per_unit)
    else:   # reflectance (TOA or SR): the reader's scale, or x1e-4 for integer products
        s = res.get("reflectance_scale") or 1.0
        sc.units = Fact("reflectance", "derived", f"x{s}" if s != 1 else "0-1")
        sc.radiance_scale = Fact(None, "derived", "reflectance: no radiance scale")
        sc._refl_scale = s
    return smp


def write_generic(sc: Scene, w: Path, dem: str | None, smp: np.ndarray) -> dict:
    L, B, S = sc.shape
    lv = sc.level.value
    # Judge bands on pixels that hold data: an ortho grid's padding is nodata in every band.
    held = smp[np.isfinite(smp).mean(1) >= 0.5]
    good = np.isfinite(held).mean(0) >= 0.95 if len(held) else np.ones(B, bool)
    sc.good_bands = Fact(good, "derived", f"{int((~good).sum())} band(s) invalid in >5% of sampled pixels dropped: "
                                          + ", ".join(f"{x:.0f}" for x in sc.wl[~good][:40]))
    wl, fw = sc.wl[good], sc.fwhm.value[good]
    # geometry, whole scene (needed whole for slope)
    if sc.location.value is not None:
        lon, lat = sc.location.value(0, L)
    else:
        # No location: a pseudo grid at 30 m spacing near (0, 0), not a constant. ISOFIT's atm_interpolation spreads
        # the superpixel atmosphere to every pixel by spatial neighbours on these coordinates; with one constant
        # coordinate every distance is 0 and it wrote AOT = H2O = elevation = 0 for the whole scene (e7 run a2).
        deg = 30.0 / 111320.0
        lon, lat = np.meshgrid(np.arange(S) * deg, -np.arange(L) * deg)
        sc.notes.append("no location: pseudo lat/lon grid at 30 m spacing near (0, 0), only for ISOFIT's spatial "
                        "interpolation of the atmosphere; not a real position")
    z, znote = (geometry.sample_dem(dem, lon, lat) if sc.location.value is not None
                else (np.zeros((L, S), np.float32), "no location: elevation 0 m assumed"))
    sc.notes.append(znote)
    sc.dem = Fact(dem if znote.startswith("DEM") else None, "metadata" if znote.startswith("DEM") else "assumed",
                  znote + ("" if znote.startswith("DEM") else "; pressure-elevation retrieval carries the surface"))
    sza, saa = sc.sun.value(0, L)
    vza, vaa = sc.view.value(0, L)
    if sc.path_length is not None:
        path_m = sc.path_length.value(0, L)
    else:
        alt_m = 1000.0 * sc.altitude_km.value
        path_m = (alt_m - z) / np.maximum(np.cos(np.radians(vza)), 0.1)
    hour = sc.time.value.hour + sc.time.value.minute / 60 + sc.time.value.second / 3600
    slope, aspect, cos_i = (geometry.slope_aspect(lat, lon, z, sza, saa) if sc.location.value is not None else
                            (np.zeros((L, S)), np.zeros((L, S)), np.cos(np.radians(sza))))
    fill = lambda a, v: np.where(np.isfinite(a), a, v)
    obs = geometry.obs_bands(fill(path_m, np.nanmedian(path_m)), fill(vaa, 0), fill(vza, 0), fill(saa, 180),
                             fill(sza, geometry.ASSUMED_SZA), slope, aspect, cos_i, hour)
    out = w / (f"{sc.fid}_rfl" if lv == "surface_reflectance" else f"{sc.fid}_rdn")
    e0 = geometry.solar_irradiance(wl, fw) if lv == "toa_reflectance" else None
    d2 = geometry.earth_sun_au(sc.time.value) ** 2
    bad_px = 0
    with open(out, "wb") as f:
        for r0 in range(0, L, ROWS):
            r1 = min(L, r0 + ROWS)
            x = sc.read_rows(r0, r1)[:, good, :].astype(np.float32)
            if lv == "radiance":
                x *= np.float32(sc.radiance_scale.value)
            elif lv == "toa_reflectance":
                mu = np.cos(np.radians(fill(sza[r0:r1], geometry.ASSUMED_SZA)))[:, None, :]
                x = (x * sc._refl_scale * e0[None, :, None] * mu / (np.pi * d2)).astype(np.float32)
            else:
                x *= np.float32(sc._refl_scale)
            dead = ~np.all(np.isfinite(x), axis=1)
            bad_px += int(dead.sum())
            if lv != "surface_reflectance":
                x = np.where(np.isfinite(x), np.maximum(x, 1e-4), NODATA)
            x[np.broadcast_to(dead[:, None, :], x.shape)] = NODATA
            f.write(np.ascontiguousarray(x, "<f4").tobytes())
    _hdr(out, S, L, len(wl), "bil", wl=wl, fwhm=fw, desc=f"{lv} from {Path(sc.path).name} (readers.prep)", sc=sc)
    loc_arr = np.stack([lon, lat, z], -1).astype("<f4")
    loc_arr.tofile(w / f"{sc.fid}_loc")
    _hdr(w / f"{sc.fid}_loc", S, L, 3, "bip", names=["longitude", "latitude", "elevation"], sc=sc)
    np.stack([np.asarray(b, np.float32) for b in obs], -1).astype("<f4").tofile(w / f"{sc.fid}_obs")
    _hdr(w / f"{sc.fid}_obs", S, L, 10, "bip", names=geometry.OBS_NAMES, sc=sc)
    np.savetxt(w / "wavelengths.txt", np.c_[np.arange(1, len(wl) + 1), wl, fw], fmt="%d\t%.4f\t%.4f")
    return {"pixels_invalid": bad_px, "bands_kept": int(good.sum()), "bands_delivered": int(B),
            "elevation_m": [float(np.nanmin(z)), float(np.nanmax(z))],
            "sun_zenith": [float(np.nanmin(sza)), float(np.nanmax(sza))],
            "view_zenith": [float(np.nanmin(vza)), float(np.nanmax(vza))]}


def _hdr(path, cols, rows, nb, inter, wl=None, fwhm=None, names=None, desc="", sc=None):
    L = ["ENVI", f"description = {{{desc}}}", f"samples = {cols}", f"lines = {rows}", f"bands = {nb}",
         "header offset = 0", "file type = ENVI Standard", "data type = 4", f"interleave = {inter}", "byte order = 0",
         f"data ignore value = {NODATA:g}"]
    if wl is not None:
        L += ["wavelength units = Nanometers", "wavelength = {" + ", ".join(f"{v:.4f}" for v in wl) + "}",
              "fwhm = {" + ", ".join(f"{v:.4f}" for v in fwhm) + "}"]
    if names:
        L.append("band names = {" + ", ".join(names) + "}")
    info = getattr(sc, "_info", None)
    if info is not None and info.transform is not None and info.crs is not None:
        t = info.transform
        L.append(f"map info = {{Arbitrary, 1, 1, {t.c}, {t.f}, {t.a}, {-t.e}, units=Meters}}")
        L.append("coordinate system string = {" + info.crs.to_wkt() + "}")
    Path(str(path) + ".hdr").write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- native paths

def native_enmap(p: Path, w: Path, dem: str) -> dict:
    import enmap_l1c_preprocess as E
    if E.main(["--input", str(p), "--dem", dem, "--workdir", str(w)]):
        raise SystemExit("enmap_l1c_preprocess failed")
    sid = Path(glob.glob(str(p / "*-METADATA.XML"))[0]).name[:-len("-METADATA.XML")]
    for k in ("rdn", "loc", "obs"):   # apply_oe's enmap branch parses the acquisition time out of the name
        os.replace(w / k, w / f"{sid}_{k}")
        os.replace(w / f"{k}.hdr", w / f"{sid}_{k}.hdr")
    return {"fid": sid, "isofit_sensor": "enmap"}


def native_prisma(p: Path, w: Path, dem: str) -> dict:
    import prisma_l1_preprocess as P
    P.main(["--l1", str(p), "--dem", dem, "--workdir", str(w)])
    return {"fid": (w / "fid.txt").read_text().strip(), "isofit_sensor": "prisma"}


def native_ang(p: Path, w: Path) -> dict:
    """AVIRIS-NG L1 as delivered: link the cubes under ISOFIT's names (no copy), trim header lists to the band count."""
    from spectral.io import envi
    rdn = Path(glob.glob(str(p / "ang*_rdn_*_img.hdr"))[0])
    fid = rdn.name.split("_")[0]
    for k, pat in (("rdn", "ang*_rdn_*_img.hdr"), ("loc", "*_loc_ort.hdr"), ("obs", "*_obs_ort.hdr")):
        h = Path(glob.glob(str(p / pat))[0])
        data = next(x for x in (h.with_suffix(".bin"), h.with_suffix(".img"), h.with_suffix("")) if x.exists())
        dst = w / f"{fid}_{k}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(data.resolve(), dst)
        md = envi.read_envi_header(str(h))
        nb = int(md["bands"])
        for key in ("fwhm", "wavelength", "bbl", "smoothing factors", "correction factors"):
            if isinstance(md.get(key), list) and len(md[key]) > nb:
                md[key] = md[key][:nb]   # v2 headers carry 480-entry fwhm lists for 425 bands
        envi.write_envi_header(str(dst) + ".hdr", md)
    md = envi.read_envi_header(str(rdn))
    wl = np.asarray(md["wavelength"], float)
    fw = np.asarray(md["fwhm"], float)[:len(wl)]
    np.savetxt(w / "wavelengths.txt", np.c_[np.arange(1, len(wl) + 1), wl, fw], fmt="%d\t%.4f\t%.4f")
    return {"fid": fid, "isofit_sensor": "ang"}


def native_record(fmt: str, w: Path, rec: dict, sensor: str) -> Scene:
    """A Scene over what a native preprocessor wrote, so the level check and scene.json work the same way."""
    fid = rec["fid"]
    info = cube.open_envi(str(w / f"{fid}_rdn.hdr"))
    t = sensors.acquisition_time(fid, {})
    meta = Fact("per pixel", "metadata", f"{fmt} product metadata (native preprocessor; its log lists any fallback)")
    sc = Scene(str(w / f"{fid}_rdn"), fmt, Fact(sensor, "metadata", "native product"), info.shape, info.interleave,
               info.wl, Fact(info.fwhm, "metadata", "product"), info.read_rows,
               Fact("uW/cm^2/sr/nm", "metadata", "converted by the native preprocessor"),
               Fact(1.0, "metadata", "native preprocessor output"), t, meta, meta, meta,
               Fact(None, "metadata", "product geometry"), fid=fid, isofit_sensor=rec["isofit_sensor"])
    sc._extra, sc._dtype, sc._info = {"level": "radiance"}, info.dtype, info
    return sc


# --------------------------------------------------------------------------- main

WIDEN_M = 3000.0


def widen_elevation(loc: Path) -> None:
    """No DEM, so every elevation is a 0 m placeholder. ISOFIT sets its pressure-elevation grid to the location cube's
    min - 2 km .. max + 2 km (template_construction.get_metadata_from_loc) and has no setting for it, so all-zero
    elevations would cap the retrieval at 2 km. One corner pixel's placeholder becomes 3 km: the grid spans 0-5 km
    and every other pixel keeps 0 m."""
    m = np.memmap(loc, dtype="<f4", mode="r+")
    m[2] = WIDEN_M   # BIP: lon, lat, elevation of pixel (0, 0)
    m.flush()


def centre_of(loc: Path) -> tuple[float, float]:
    from spectral.io import envi
    m = envi.open(str(loc) + ".hdr", str(loc)).open_memmap(interleave="bip")[::10, ::10, :2].astype(float)
    m[m <= -9990] = np.nan
    return float(np.nanmedian(m[..., 1])), float(np.nanmedian(m[..., 0]))


def flat_dem(w: Path) -> str:
    """A global 0 m DEM (0.1 deg, EPSG:4326) for a native product delivered without one: the native preprocessors
    need a raster to sample. Flagged as assumed; ISOFIT's pressure-elevation retrieval then carries the surface."""
    import rasterio
    from rasterio.transform import from_origin
    p = w / "dem_flat_0m.tif"
    with rasterio.open(p, "w", driver="GTiff", width=3600, height=1800, count=1, dtype="int16", crs="EPSG:4326",
                       transform=from_origin(-180, 90, 0.1, 0.1), compress="deflate") as d:
        d.write(np.zeros((1800, 3600), np.int16), 1)
    return str(p)


def season(t: datetime, lat: float | None) -> str:
    north = lat is None or lat >= 0
    summer = 4 <= t.month <= 9
    return "summer" if summer == north else "winter"


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--input", required=True, type=Path)
    a.add_argument("--workdir", required=True, type=Path)
    a.add_argument("--dem", default=os.environ.get("DEM_VRT"))
    a.add_argument("--sensor", choices=sorted(sensors.SENSORS))
    a.add_argument("--input-level", choices=("radiance", "toa", "sr"))
    a.add_argument("--radiance-scale", type=float, help="factor from the delivered numbers to uW/cm^2/sr/nm")
    a.add_argument("--wavelength-file", help="band centres (and FWHM) when the delivery carries none: 1, 2 or 3 "
                                             "columns (wl | wl fwhm | index wl fwhm), nm or um, one row per band")
    o = a.parse_args(argv)
    w = o.workdir
    w.mkdir(parents=True, exist_ok=True)
    fmt, path = detect(o.input)
    log.info("input %s: %s", path, fmt)
    stats = {}
    if fmt in ("enmap_l1c", "prisma_l1", "aviris_ng_l1"):
        dem = o.dem if o.dem and glob.glob(o.dem) else None
        if fmt != "aviris_ng_l1" and dem is None:
            log.warning("no DEM (%r): using a flat 0 m DEM, flagged", o.dem)
            dem = flat_dem(w)
        rec = (native_enmap(path, w, dem) if fmt == "enmap_l1c" else native_prisma(path, w, dem)
               if fmt == "prisma_l1" else native_ang(path, w))
        sc = native_record(fmt, w, rec, {"enmap_l1c": "enmap", "prisma_l1": "prisma", "aviris_ng_l1": "aviris_ng"}[fmt])
        sc.dem = (Fact(None, "metadata", "elevation from the product's own location cube (loc_ort)")
                  if fmt == "aviris_ng_l1" else
                  Fact(dem, "assumed", "no DEM given: flat 0 m; pressure-elevation retrieval carries the surface")
                  if dem.endswith("dem_flat_0m.tif") else Fact(dem, "metadata", f"DEM {dem}"))
        decide_level(sc, o.input_level, None)
        if sc.level.value != "radiance":
            sc.notes.append(f"WARNING: native L1 product but the level tests say {sc.level_tests['level']}")
    else:
        sc = build(fmt, path, o.sensor, o.wavelength_file)
        smp = decide_level(sc, o.input_level, o.radiance_scale)
        stats = write_generic(sc, w, o.dem, smp)
        lat_mid = None if sc.location.value is None else float(np.nanmedian(sc.location.value(sc.lines // 2, sc.lines // 2 + 1)[1]))
        (w / "season.txt").write_text(season(sc.time.value, lat_mid))
    (w / "fid.txt").write_text(sc.fid)
    if sc.dem.source == "assumed" and fmt != "aviris_ng_l1":
        widen_elevation(w / f"{sc.fid}_loc")
        sc.notes.append(f"pressure-elevation grid widened to 0-{(WIDEN_M / 1000) + 2:.0f} km: pixel (0, 0) "
                        f"placeholder elevation {WIDEN_M:.0f} m, all others 0 m (no DEM)")
    lat, lon = centre_of(w / f"{sc.fid}_loc") if sc.location.value is not None else (None, None)
    t = getattr(sc._info, "transform", None)
    px_m = (None if t is None else abs(t.a) if sc._info.crs is not None and sc._info.crs.is_projected
            else abs(t.a) * 111320.0 * np.cos(np.radians(lat or 0)))
    px_m = round(float(px_m), 2) if px_m else 30.0   # swath products (PRISMA, Tanager basic) are 30 m
    import atm_post   # the per-sensor rules table (noise, LUT, AOT prior, post-processing, harmonisation)
    rule = atm_post.rule_name(sc.sensor.value)
    rules = atm_post.SENSOR_RULES[rule]
    # EnMAP / PRISMA run ISOFIT's generic noise model by design: that is the validated configuration (AC26-C45,
    # AC26-ACC), not a gap. Only sensors never validated (ghost, generic) count it as an assumption.
    sc.noise = (Fact(rules.instrument_noise, "metadata", f"per-sensor noise model ({rule} rule)")
                if rules.instrument_noise else
                Fact(None, "metadata", f"ISOFIT's generic noise model, as validated for {rule}")
                if rule in ("enmap", "prisma") else
                Fact(None, "assumed", f"ISOFIT's generic noise model: no noise file for the {rule} rule"))
    rec = {"input": str(o.input), "format": fmt, "interleave": sc.interleave, "fid": sc.fid,
           "isofit_sensor": sc.isofit_sensor, "rule": rule, "rules": atm_post.asdict(rules),
           "centre_lat": lat, "centre_lon": lon, "date": f"{sc.time.value:%Y%m%d}", "pixel_m": px_m,
           "season": (w / "season.txt").read_text().strip() if (w / "season.txt").exists() else "summer",
           "lines": sc.lines, "bands": sc.shape[1], "samples": sc.samples,
           "wavelength_nm": [float(sc.wl[0]), float(sc.wl[-1])], "level": sc.level.value,
           "pass_through": sc.level.value == "surface_reflectance", "level_tests": sc.level_tests,
           "grade": sc.grade(), "checklist": sc.checklist(), "notes": sc.notes, "stats": stats,
           "written": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (w / "scene.json").write_text(json.dumps(rec, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)))
    log.info("grade %s, level %s (%s), sensor %s, isofit sensor %s", rec["grade"], sc.level.value,
             sc.level.note, sc.sensor.value, sc.isofit_sensor)
    for r in rec["checklist"]:
        log.info("  %-14s %-9s %s", r["item"], r["source"], r["note"])
    for n in sc.notes:
        log.info("  note: %s", n)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
