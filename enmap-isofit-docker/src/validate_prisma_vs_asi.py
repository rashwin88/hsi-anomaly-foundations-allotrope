"""
Compare an L1 -> L2 run on PRISMA against ASI's own L2D of the same acquisition.

Unlike EnMAP, the two products do not share a grid: ASI's L2D is resampled to
UTM with ground control points, while ours stays in L1 swath geometry, whose
lat/lon can be off by a few hundred metres (L2D attribute
PAN_HYP_ACT_RESIDUAL_m reached 229 m on the 2024-05-11 scene). So:

  1. both lat/lon grids go to the L2D's UTM zone (root attr Epsg_Code);
  2. every L2D pixel takes the nearest of our pixels within 30 m;
  3. a shift field is estimated by phase correlation on an 860 nm band, per
     tile of a 6 x 6 grid, and the lookup is redone with our coordinates
     moved by it;
  4. the matched cubes go through validate_vs_paco.compare_class, so the
     numbers mean the same as in the EnMAP table.

ASI publishes no land/water map; water here is rho(860) < 0.05 and
rho(560) > rho(860), taken from the ASI side.

Our side is read from an ACOLITE L2R NetCDF (lat, lon, rhos_* with a
`wavelength` attribute). Bands are paired by nearest wavelength within 1 nm.

  python validate_prisma_vs_asi.py --asi <L2D.he5> --ours <L2R.nc> --csv <out.csv>
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import h5py
import netCDF4
import numpy as np
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

from validate_vs_paco import compare_class, print_tables, write_csv

log = logging.getLogger("validate_prisma_vs_asi")

SWATH = "HDFEOS/SWATHS/PRS_L2D_HCO"
PIXEL_M = 30.0
MIN_PIXELS = 500
TILES = 6


def read_asi(path: Path):
    """-> (wavelengths ascending, cube (bands, rows, cols) float32 NaN=nodata, lat, lon, epsg)."""
    with h5py.File(path, "r") as f:
        a = f.attrs
        parts = []
        for fam in ("Vnir", "Swir"):
            wl = np.asarray(a[f"List_Cw_{fam}"], float)
            keep = np.asarray(a[f"List_Cw_{fam}_Flags"]).astype(bool) & (wl > 0)
            lo, hi = float(a[f"L2Scale{fam}Min"]), float(a[f"L2Scale{fam}Max"])
            dn = f[f"{SWATH}/Data Fields/{fam.upper()}_Cube"][:, keep, :]
            r = np.where(dn == 0, np.nan, lo + dn.astype("float32") * (hi - lo) / 65535)
            parts.append((wl[keep], r.transpose(1, 0, 2)))
        lat = f[f"{SWATH}/Geolocation Fields/Latitude"][:]
        lon = f[f"{SWATH}/Geolocation Fields/Longitude"][:]
        epsg = int(a["Epsg_Code"])
    wl = np.concatenate([p[0] for p in parts])
    cube = np.concatenate([p[1] for p in parts])
    order = np.argsort(wl)
    return wl[order], cube[order], lat, lon, epsg


def read_ours(path: Path):
    """-> (wavelengths ascending, cube (bands, rows, cols), lat, lon)."""
    ds = netCDF4.Dataset(path)
    bands = []
    for name, var in ds.variables.items():
        if name.startswith("rhos_"):
            bands.append((float(var.getncattr("wavelength")),
                          np.ma.filled(var[:].astype("float32"), np.nan)))
    bands.sort(key=lambda b: b[0])
    lat = np.ma.filled(ds.variables["lat"][:], np.nan)
    lon = np.ma.filled(ds.variables["lon"][:], np.nan)
    return np.array([b[0] for b in bands]), np.stack([b[1] for b in bands]), lat, lon


def lookup(tree: cKDTree, xy: np.ndarray, n: int) -> np.ndarray:
    """Index into our flattened pixels for each target point; n where no match."""
    _, idx = tree.query(xy, distance_upper_bound=PIXEL_M)
    return np.where(idx < n, idx, n)


def resample(cube: np.ndarray, idx: np.ndarray, shape) -> np.ndarray:
    flat = cube.reshape(cube.shape[0], -1)
    flat = np.concatenate([flat, np.full((flat.shape[0], 1), np.nan, flat.dtype)], axis=1)
    return flat[:, idx].reshape(cube.shape[0], *shape)


def shift_field(ref: np.ndarray, moving: np.ndarray, tiles: int = TILES):
    """Per-pixel (rows, cols) shift that registers `moving` onto `ref`.

    The L1 geolocation error is not rigid - on 2024-05-11 it runs from ~3
    columns at the swath edges to ~5 mid-swath - so a shift is measured per
    tile and bilinearly spread over the image. Tiles that are mostly empty,
    or disagree with the median by more than 3 pixels, take the median.
    """
    H, W = ref.shape
    th, tw = H // tiles, W // tiles
    est = np.full((tiles, tiles, 2), np.nan)
    for i in range(tiles):
        for j in range(tiles):
            a = ref[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            o = moving[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if (a > 0).mean() > 0.5 and (o > 0).mean() > 0.5:
                est[i, j] = phase_cross_correlation(a, o, upsample_factor=10)[0]
    med = np.nanmedian(est.reshape(-1, 2), axis=0)
    bad = ~np.isfinite(est).all(-1) | (np.abs(est - med) > 3).any(-1)
    est[bad] = med
    centres_r = (np.arange(tiles) + 0.5) * th
    centres_c = (np.arange(tiles) + 0.5) * tw
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = []
    for k in range(2):
        f = RegularGridInterpolator((centres_r, centres_c), est[..., k],
                                    bounds_error=False, fill_value=None)
        out.append(f(np.c_[rr.ravel(), cc.ravel()]).reshape(H, W))
    return out[0], out[1]


def compare_scene(name: str, asi_path: Path, ours_path: Path):
    wl_a, A, lat_a, lon_a, epsg = read_asi(asi_path)
    wl_o, O, lat_o, lon_o = read_ours(ours_path)
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    xa, ya = to_utm.transform(lon_a.ravel(), lat_a.ravel())
    xo, yo = to_utm.transform(lon_o.ravel(), lat_o.ravel())
    ok = np.isfinite(xo) & np.isfinite(yo)
    src = np.flatnonzero(ok)
    tree = cKDTree(np.c_[xo[ok], yo[ok]])
    n = len(src)

    # Nearest wavelength within 1 nm; ASI bands with no partner stay as NaN
    # rows, so the band axis is ASI's and statistics skip the gaps.
    j = np.abs(wl_o[None, :] - wl_a[:, None]).argmin(1)
    pair = np.abs(wl_o[j] - wl_a) <= 1.0
    wl = wl_a
    O = O.reshape(O.shape[0], -1)[:, src][j]
    O[~pair] = np.nan
    O = O.reshape(O.shape[0], 1, -1)

    def on_asi_grid(dx=0.0, dy=0.0):
        idx = lookup(tree, np.c_[xa - np.ravel(dx), ya - np.ravel(dy)], n)
        return resample(O, idx, lat_a.shape)

    ref = int(np.argmin(np.abs(wl - 860)))
    first = on_asi_grid()
    rows, cols = shift_field(np.nan_to_num(A[ref]), np.nan_to_num(first[ref]))
    # A shift of +c columns means our pixel belongs c columns further east;
    # rows run north -> south, so +r rows is r pixels further south.
    dx, dy = cols * PIXEL_M, -rows * PIXEL_M
    log.info("%s: %d of %d ASI bands paired, geolocation shift dx %+.0f..%+.0f m, "
             "dy %+.0f..%+.0f m", name, pair.sum(), len(wl), dx.min(), dx.max(),
             dy.min(), dy.max())
    Og = on_asi_grid(dx, dy)
    dx, dy = float(np.median(dx)), float(np.median(dy))

    r860, r560 = A[ref], A[int(np.argmin(np.abs(wl - 560)))]
    water = (r860 < 0.05) & (r560 > r860)
    bands, summaries = [], []
    for cls, m in (("land", ~water & np.isfinite(r860)), ("water", water)):
        if m.sum() < MIN_PIXELS:
            continue
        b, s = compare_class(name, cls, A[:, m], Og[:, m], wl)
        s["shift_dx_m"], s["shift_dy_m"] = dx, dy
        bands += b
        summaries.append(s)
    return bands, summaries


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--asi", required=True, type=Path, nargs="+", help="ASI L2D .he5 files")
    p.add_argument("--ours", required=True, type=Path, nargs="+", help="L2R .nc files, same order")
    p.add_argument("--csv", help="Write the summary here; per-band rows go to <name>_bands.csv")
    args = p.parse_args(argv)
    if len(args.asi) != len(args.ours):
        log.error("--asi and --ours need the same number of files")
        return 1
    bands, summaries = [], []
    for asi, ours in zip(args.asi, args.ours):
        name = re.search(r"_(\d{14})_", asi.name).group(1)
        log.info("Comparing %s", name)
        b, s = compare_scene(name, asi, ours)
        bands += b
        summaries += s
    print_tables(bands, summaries)
    if args.csv and summaries:
        out = Path(args.csv)
        write_csv(summaries, out)
        write_csv(bands, out.with_name(out.stem + "_bands.csv"))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
