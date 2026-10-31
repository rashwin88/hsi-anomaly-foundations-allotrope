"""Final reflectance -> the submission product: GeoTIFF and ENVI BSQ, int16 x10000, water bands dropped.

  python -m writers --rfl <ENVI reflectance> --scene <work>/scene.json --out <dir>
                    [--report atm_post.json] [--formats tif,bsq] [--no-compress]

What the organisers asked for (mail 2026-10-31): .geotiff / .bsq with all wavelength information, reflectance 0-1 or
x10000, water absorption bands removed. So, for every sensor:
  - bands outside ISOFIT's inversion windows (rfl_qc.INVERSION_WINDOWS_NM: 1360-1410 and 1800-1970 nm, and beyond
    2500 nm) are dropped, and so is any band that holds no valid pixel; the dropped list goes into the metadata;
  - int16 = round(reflectance x 10000), nodata -9999 (values below -0.2 were already nodata, rfl_qc; values above
    3.2767 saturate at 32767, and rfl_qc flags anything above 2.5);
  - <name>.tif: tiled GeoTIFF, DEFLATE + horizontal predictor on all cores, band descriptions "<wl> nm", per-band
    wavelength / fwhm tags, scale and units in the dataset tags;
  - <name>.bsq + <name>.hdr: written by GDAL's ENVI driver, so the map info (rotation included) and the CRS are
    GDAL's own; wavelength, fwhm, band names, data ignore value and reflectance scale factor are added to the header.
Georeference: the input's own (EnMAP L1C image, AVIRIS-NG map info, Tanager ortho grid, any CRS-bearing GeoTIFF or
ENVI). Swath inputs (PRISMA L1, Tanager basic) have none: the GeoTIFF gets lat/lon tie points (GCPs, every 50 pixels)
from the location cube, the ENVI header gets them as geo points (GDAL keeps 4), and <name>_latlon.tif carries the
per-pixel latitude / longitude for both; the data stays on the L1 / L2C swath grid (no resampling).
Sidecars: <name>_outlier.tif and <name>_snow.tif (rfl_qc masks) when present, and <name>_provenance.json (scene
checklist and grade, rule, retrieved atmosphere, harmonisation, dropped bands, write times).
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

import rfl_qc

log = logging.getLogger("writers")
NODATA = -9999
SCALE = 10000.0
GCP_STEP = 50


def keep_bands(wl: np.ndarray, valid_any: np.ndarray) -> np.ndarray:
    inside = np.any([(wl >= lo) & (wl <= hi) for lo, hi in rfl_qc.INVERSION_WINDOWS_NM], axis=0)
    return inside & valid_any


def georef(scene: dict, work: Path, fid: str, shape: tuple[int, int]):
    """(transform, crs, gcps) for the product grid; exactly one of transform / gcps is set (or none)."""
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.crs import CRS
    src, fmt = scene["input"], scene["format"]
    cand = []
    if fmt == "enmap_l1c":
        cand = [p for p in sorted(Path(src).glob("*SPECTRAL_IMAGE*")) if p.suffix.upper() in (".TIF", ".TIFF")]
    elif fmt == "aviris_ng_l1":
        cand = sorted(Path(src).glob("ang*_rdn_*_img.bin")) + sorted(Path(src).glob("ang*_rdn_*_img"))
    elif fmt in ("envi", "geotiff"):
        cand = [Path(src[:-4])] if src.lower().endswith(".hdr") else [Path(src)]
        cand += [Path(src[:-4] + e) for e in (".dat", ".img", ".bin", ".bsq", ".bil", ".bip") if src.lower().endswith(".hdr")]
    elif fmt == "tanager_h5":
        from readers.tanager import open_tanager
        info, _ = open_tanager(src)
        if info.transform is not None:
            return info.transform, info.crs, None
    for c in cand:
        if not c.exists():
            continue
        try:
            with rasterio.open(c) as d:
                if d.crs is not None and not d.transform.is_identity and (d.height, d.width) == shape:
                    return d.transform, d.crs, None
        except Exception:   # noqa: BLE001 - not GDAL-readable: try the next candidate
            continue
    if scene.get("centre_lat") is None:
        return None, None, None   # no location at all: no georeference, said so in provenance
    from spectral.io import envi
    loc = envi.open(str(work / f"{fid}_loc.hdr"), str(work / f"{fid}_loc")).open_memmap(interleave="bip")
    gcps = []
    rows, cols = shape
    for r in list(range(0, rows, GCP_STEP)) + [rows - 1]:
        for c in list(range(0, cols, GCP_STEP)) + [cols - 1]:
            lon, lat = float(loc[r, c, 0]), float(loc[r, c, 1])
            if -180 <= lon <= 180 and -90 <= lat <= 90 and (lon, lat) != (0.0, 0.0):
                gcps.append(GroundControlPoint(row=r + 0.5, col=c + 0.5, x=lon, y=lat))
    return None, CRS.from_epsg(4326), gcps or None


def read_scaled(rfl: str):
    """(int16 cube (bands, rows, cols), wavelengths, fwhm). Reads the BIL / BSQ / BIP reflectance in row blocks."""
    from spectral.io import envi
    img = envi.open(rfl + ".hdr", rfl)
    md = img.metadata
    wl = np.asarray(md["wavelength"], float)
    wl = wl * 1000 if wl.max() < 100 else wl
    fw = np.asarray(md.get("fwhm", np.zeros(len(wl))), float)[:len(wl)]
    fw = fw * 1000 if fw.size and fw.max() < 1 else fw
    m = img.open_memmap(interleave="bip")
    L, S, B = m.shape
    out = np.empty((B, L, S), np.int16)
    for r0 in range(0, L, 128):
        x = np.asarray(m[r0:r0 + 128], np.float32).transpose(2, 0, 1)
        ok = np.isfinite(x) & (x > -9990)
        y = np.clip(np.rint(x * SCALE), -9998, 32767)
        out[:, r0:r0 + 128, :] = np.where(ok, y, NODATA).astype(np.int16)
    return out, wl, fw


def write(rfl: str, scene_path: str, out_dir: str, report: str | None, formats=("tif", "bsq"), compress=True) -> dict:
    import rasterio
    t0 = time.perf_counter()
    scene = json.loads(Path(scene_path).read_text())
    work, fid = Path(scene_path).parent, scene["fid"]
    cube, wl, fw = read_scaled(rfl)
    valid_any = (cube != NODATA).reshape(cube.shape[0], -1).any(1)
    keep = keep_bands(wl, valid_any)
    cube, wl_k, fw_k = cube[keep], wl[keep], fw[keep]
    B, L, S = cube.shape
    transform, crs, gcps = georef(scene, work, fid, (L, S))
    t_read = time.perf_counter() - t0
    name = f"{fid}_L2_SR"
    os.makedirs(out_dir, exist_ok=True)
    common = dict(width=S, height=L, count=B, dtype="int16", nodata=NODATA)
    if transform is not None:
        common.update(transform=transform, crs=crs)
    files, times = {}, {"read_scale_s": round(t_read, 1)}
    if "tif" in formats:
        t = time.perf_counter()
        p = Path(out_dir) / f"{name}.tif"
        opts = dict(tiled=True, blockxsize=256, blockysize=256, interleave="band", BIGTIFF="IF_SAFER")
        if compress:
            opts.update(compress="deflate", predictor=2, num_threads="all_cpus")
        with rasterio.open(p, "w", driver="GTiff", **common, **opts) as d:
            if gcps:
                d.gcps = (gcps, crs)
            d.write(cube)
            for i in range(B):
                d.set_band_description(i + 1, f"{wl_k[i]:.2f} nm")
                d.update_tags(i + 1, wavelength=f"{wl_k[i]:.4f}", fwhm=f"{fw_k[i]:.4f}", wavelength_units="nm")
            d.update_tags(product="surface reflectance", scale_factor=f"{SCALE:g}", nodata=str(NODATA),
                          wavelength_units="nm", bands_dropped=",".join(f"{w:.1f}" for w in wl[~keep]),
                          sensor=scene.get("rule", ""), grade=scene.get("grade", ""),
                          georeference="map" if transform is not None else "gcps" if gcps else "none")
        files["tif"], times["tif_s"] = str(p), round(time.perf_counter() - t, 1)
    if "bsq" in formats:
        t = time.perf_counter()
        p = Path(out_dir) / f"{name}.bsq"
        with rasterio.open(p, "w", driver="ENVI", interleave="bsq", **common) as d:
            if gcps:
                d.gcps = (gcps, crs)
            d.write(cube)
        hdr = Path(out_dir) / f"{name}.hdr"
        _augment_hdr(hdr, wl_k, fw_k)
        for junk in (Path(str(p) + ".aux.xml"),):
            if junk.exists():
                junk.unlink()
        files["bsq"], files["hdr"], times["bsq_s"] = str(p), str(hdr), round(time.perf_counter() - t, 1)
    files.update(_sidecars(rfl, out_dir, name, common, gcps, crs))
    if gcps:   # GDAL's ENVI writer keeps only 4 tie points: give both formats the per-pixel lat/lon of the swath
        from spectral.io import envi
        loc = envi.open(str(work / f"{fid}_loc.hdr"), str(work / f"{fid}_loc")).open_memmap(interleave="bip")
        p = Path(out_dir) / f"{name}_latlon.tif"
        with rasterio.open(p, "w", driver="GTiff", width=S, height=L, count=2, dtype="float32", compress="deflate",
                           predictor=3, tiled=True) as d:
            d.write(np.stack([loc[..., 1], loc[..., 0]]).astype(np.float32))
            d.set_band_description(1, "latitude (deg, WGS-84)")
            d.set_band_description(2, "longitude (deg, WGS-84)")
        files["latlon"] = str(p)
    prov = {"product": files, "scale_factor": SCALE, "nodata": NODATA, "bands_kept": int(B),
            "bands_dropped_nm": [round(float(w), 2) for w in wl[~keep]],
            "georeference": "map" if transform is not None else "gcps (swath, lat/lon tie points)" if gcps else
            "none (no location in the delivery)", "write_times_s": times, "scene": scene}
    if report and Path(report).exists():
        prov["atm_post"] = json.loads(Path(report).read_text())
    hj = Path(rfl + ".json")
    if rfl.endswith("_harm") and hj.exists():
        prov["harmonisation"] = json.loads(hj.read_text())
    Path(out_dir, f"{name}_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    log.info("wrote %s: %d bands kept, %d dropped, georeference %s, times %s", name, B, int((~keep).sum()),
             prov["georeference"], times)
    return prov


def _augment_hdr(hdr: Path, wl, fw) -> None:
    from spectral.io import envi
    md = envi.read_envi_header(str(hdr))
    md.update({"wavelength units": "Nanometers", "wavelength": [f"{w:.4f}" for w in wl], "fwhm": [f"{f:.4f}" for f in fw],
               "band names": [f"{w:.2f} nm" for w in wl], "data ignore value": str(NODATA),
               "reflectance scale factor": f"{SCALE:g}",
               "description": "surface reflectance x10000 (int16), water absorption bands dropped"})
    envi.write_envi_header(str(hdr), md)


def _sidecars(rfl: str, out_dir: str, name: str, common: dict, gcps, crs) -> dict:
    """rfl_qc's uint8 masks (outlier, snow) as small GeoTIFFs on the same grid."""
    import rasterio
    from spectral.io import envi
    base = rfl[:-5] if rfl.endswith("_harm") else rfl
    out = {}
    for tag in ("outlier", "snow"):
        src = f"{base}_{tag}"
        if not Path(src + ".hdr").exists():
            continue
        m = np.asarray(envi.open(src + ".hdr", src).open_memmap(interleave="bip")[..., 0], np.uint8)
        p = Path(out_dir) / f"{name}_{tag}.tif"
        kw = {k: v for k, v in common.items() if k not in ("count", "dtype", "nodata")}
        with rasterio.open(p, "w", driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate", **kw) as d:
            if gcps:
                d.gcps = (gcps, crs)
            d.write(m, 1)
        out[tag] = str(p)
    return out


def main(argv=None) -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--rfl", required=True)
    a.add_argument("--scene", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--report")
    a.add_argument("--formats", default="tif,bsq")
    a.add_argument("--no-compress", action="store_true")
    o = a.parse_args(argv)
    write(o.rfl, o.scene, o.out, o.report, tuple(o.formats.split(",")), not o.no_compress)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
