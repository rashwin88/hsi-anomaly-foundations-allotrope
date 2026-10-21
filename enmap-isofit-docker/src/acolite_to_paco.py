"""
Repack an ACOLITE L2R NetCDF of an EnMAP scene as a PACO-layout SPECTRAL_IMAGE.TIF.

Only the reflectance cube is written - enough for validate_vs_paco.py, which
takes land/water classes from the PACO side. Georeferencing comes from the
L1C SPECTRAL_IMAGE, which shares its grid with both L2 products.

Bands are matched to the L1C band list by the `wavelength` attribute, not by
variable name: ACOLITE names bands `rhos_<rounded nm>`, so the VNIR/SWIR
overlap pairs (911.6/912.0 and 993.1/993.3 nm) collide and one of each pair
is overwritten. Those bands, and every band ACOLITE skips because gas
transmittance is below `min_tgas_rho`, are written as nodata.

Run inside the ACOLITE image (netCDF4 + rasterio):

  python acolite_to_paco.py --l2r <L2R.nc> --l1c <L1C product dir> --out <dir>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import netCDF4
import numpy as np
import rasterio

log = logging.getLogger("acolite_to_paco")

NODATA = -32768


def convert(l2r: Path, l1c_dir: Path, out_dir: Path) -> Path:
    l1c = next(l1c_dir.glob("ENMAP01-____L1C-DT*-SPECTRAL_IMAGE.TIF"))
    ds = netCDF4.Dataset(l2r)
    waves = np.asarray(ds.getncattr("band_waves"), dtype=float)
    with rasterio.open(l1c) as src:
        profile = src.profile
    if profile["count"] != len(waves):
        raise ValueError(f"L1C has {profile['count']} bands, ACOLITE saw {len(waves)}")

    cube = np.full((len(waves), profile["height"], profile["width"]), NODATA, np.int16)
    filled = 0
    for name, var in ds.variables.items():
        if not name.startswith("rhos_"):
            continue
        i = int(np.argmin(np.abs(waves - float(var.getncattr("wavelength")))))
        a = np.ma.filled(var[:].astype("float32"), np.nan)
        ok = np.isfinite(a)
        cube[i][ok] = np.clip(np.round(a[ok] * 1e4), -32767, 32767).astype(np.int16)
        filled += 1
    log.info("%s: %d of %d bands carry reflectance (AOT550 %.3f, %s)", l2r.name, filled,
             len(waves), float(ds.getncattr("ac_aot_550")), ds.getncattr("ac_model"))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / l1c.name.replace("____L1C-", "____L2A-")
    profile.update(dtype="int16", nodata=NODATA, compress="deflate", predictor=2,
                   tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(cube)
    log.info("Wrote %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l2r", required=True, type=Path)
    p.add_argument("--l1c", required=True, type=Path, help="EnMAP L1C product dir")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)
    convert(args.l2r, args.l1c, args.out)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.exit(main())
