"""
Write a surface-reflectance cube as the AIGC Level-1 deliverable: GeoTIFF and/or ENVI BSQ + .hdr, every wavelength
recorded, water-absorption bands removed, reflectance x 10000 as int16 (nodata -9999).

    python scripts/write_reflectance_product.py --cube <reflectance> --out <stem> [--wavelengths wl.json]
        [--fwhm fwhm.json] [--input-scale 1|10000] [--georef-from <GeoTIFF / ENVI / PRISMA .he5>]
        [--drop 1340-1460,1790-1960] [--format both|tif|bsq]

--cube is anything rasterio reads (GeoTIFF, ENVI .bsq/.img with its .hdr - wavelengths and fwhm come from the header
when present) or a .npy (bands, rows, cols). --wavelengths / --fwhm: a JSON list or a text file with one value per
line, in the cube's band order. --input-scale: the cube's reflectance scale (1 for 0-1, 10000 for x10000).
--georef-from: where the map grid comes from when the cube has none (a raster with a CRS, or a PRISMA L2D/L1 .he5
through app/georef's product UTM grid); without it, the cube's own CRS / transform are kept. The atmospheric
correction itself is not here - it is the atm-correction module's job; this only writes its result in the format
the evaluators asked for (circular, 2026-10-31).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NODATA = -9999
SCALE = 10000


def _values(path: str | None):
    if not path:
        return None
    text = Path(path).read_text()
    try:
        return [float(v) for v in json.loads(text)]
    except ValueError:
        return [float(v) for v in text.split()]


def _envi_list(tags: dict, key: str):
    v = tags.get(key)
    return None if v is None else [float(x) for x in v.strip("{} ").split(",") if x.strip()]


def read_cube(args):
    """(cube (B, H, W) float32 reflectance 0-1, wavelengths, fwhm or None, crs, transform)."""
    import rasterio
    crs = transform = None
    wl, fwhm = _values(args.wavelengths), _values(args.fwhm)
    if args.cube.endswith(".npy"):
        cube = np.load(args.cube).astype(np.float32)
    else:
        with rasterio.open(args.cube) as src:
            cube = src.read().astype(np.float32)
            crs, transform = src.crs, src.transform
            envi = src.tags(ns="ENVI")
            wl = wl or _envi_list(envi, "wavelength")
            fwhm = fwhm or _envi_list(envi, "fwhm")
            if src.nodata is not None:
                cube[cube == src.nodata] = np.nan
    if wl is None or len(wl) != cube.shape[0]:
        raise SystemExit(f"need {cube.shape[0]} wavelengths (--wavelengths or an ENVI header), got "
                         f"{None if wl is None else len(wl)}")
    if fwhm is not None and len(fwhm) != len(wl):
        raise SystemExit(f"fwhm has {len(fwhm)} values for {len(wl)} bands")
    if args.georef_from:
        crs, transform = georef_from(args.georef_from, cube.shape[1:])
    return cube / args.input_scale, np.asarray(wl, float), (None if fwhm is None else np.asarray(fwhm, float)), crs, transform


def georef_from(path: str, shape):
    import rasterio
    from rasterio.crs import CRS
    if path.lower().endswith(".he5"):
        import h5py
        from app.georef.from_scene_file import prisma_utm_grid
        with h5py.File(path, "r") as f:
            lat = next((f[g]["Latitude"].shape for g in ("HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields",
                        "HDFEOS/SWATHS/PRS_L1_HCO/Geolocation Fields") if g in f), None)
            if lat is not None and tuple(lat) != tuple(shape):
                raise SystemExit(f"{path}: product grid {lat} vs cube {tuple(shape)}")
            grid = prisma_utm_grid(dict(f.attrs), shape)
        if grid is None:
            raise SystemExit(f"{path}: no Epsg_Code / Product_*corner attributes - give a georeferenced raster")
        return CRS.from_epsg(grid[1]), grid[0]
    with rasterio.open(path) as src:
        if (src.height, src.width) != tuple(shape):
            raise SystemExit(f"{path}: grid {src.height}x{src.width} vs cube {shape}")
        return src.crs, src.transform


def drop_mask(wl: np.ndarray, spec: str) -> np.ndarray:
    keep = np.ones(len(wl), bool)
    for part in filter(None, spec.split(",")):
        lo, hi = (float(v) for v in part.split("-"))
        keep &= ~((wl >= lo) & (wl <= hi))
    return keep


def write(cube, wl, fwhm, crs, transform, out: Path, fmt: str):
    import rasterio
    order = np.argsort(wl)
    cube, wl = cube[order], wl[order]
    fwhm = None if fwhm is None else fwhm[order]
    data = np.where(np.isfinite(cube), np.clip(np.round(cube * SCALE), -9998, 32767), NODATA).astype(np.int16)
    B, H, W = data.shape
    meta = {"wavelength units": "Nanometers", "reflectance scale factor": str(SCALE), "data ignore value": str(NODATA),
            "wavelength": "{" + ", ".join(f"{w:.3f}" for w in wl) + "}",
            "band names": "{" + ", ".join(f"{w:.2f} nm" for w in wl) + "}"}
    if fwhm is not None:
        meta["fwhm"] = "{" + ", ".join(f"{v:.3f}" for v in fwhm) + "}"
    written = []
    for kind in (("tif", "GTiff"), ("bsq", "ENVI")):
        if fmt not in ("both", kind[0]):
            continue
        path = out.with_suffix(f".{kind[0]}")
        opts = dict(compress="deflate", predictor=2, tiled=True, interleave="band") if kind[1] == "GTiff" \
            else dict(interleave="bsq")
        with rasterio.open(path, "w", driver=kind[1], height=H, width=W, count=B, dtype="int16", nodata=NODATA,
                           crs=crs, transform=transform, **opts) as dst:
            dst.write(data)
            for i, w in enumerate(wl, start=1):
                dst.set_band_description(i, f"{w:.2f} nm")
                dst.update_tags(i, wavelength=f"{w:.3f}", wavelength_units="nm",
                                **({"fwhm": f"{fwhm[i - 1]:.3f}"} if fwhm is not None else {}))
            dst.update_tags(ns="ENVI", **meta)
            dst.update_tags(reflectance_scale_factor=str(SCALE), wavelength_units="Nanometers")
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--cube", required=True)
    ap.add_argument("--out", required=True, help="output path stem (extension is replaced)")
    ap.add_argument("--wavelengths")
    ap.add_argument("--fwhm")
    ap.add_argument("--input-scale", type=float, default=1.0)
    ap.add_argument("--georef-from")
    ap.add_argument("--drop", default="1340-1460,1790-1960")
    ap.add_argument("--format", choices=("both", "tif", "bsq"), default="both")
    args = ap.parse_args()
    cube, wl, fwhm, crs, transform = read_cube(args)
    if crs is None:
        print("warning: no CRS - give --georef-from, or the product will not overlay the reference", file=sys.stderr)
    keep = drop_mask(wl, args.drop)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    paths = write(cube[keep], wl[keep], None if fwhm is None else fwhm[keep], crs, transform, out, args.format)
    print(f"{int(keep.sum())} of {len(wl)} bands kept ({len(wl) - int(keep.sum())} water bands dropped); wrote "
          + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
