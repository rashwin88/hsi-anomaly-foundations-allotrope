"""
Compare this pipeline's L2A against DLR's PACO L2A for the same scenes.

Scenes are paired by EnMAP datatake id (DTxxxxxxxxxx). For each pair, pixels
are split with PACO's own QL_QUALITY_CLASSES (1 = land, 2 = water) and
compared band by band:

  * per-band table  median PACO, median ours, median (ours - PACO), Pearson r
  * summary row     mean |ours - PACO| over all bands, over 460-2450 nm (the
                    Allotrope common grid), below 700 nm and above 800 nm;
                    median bias at 460 nm; median spectral angle (degrees)

Only pixels valid in both products count. Bands PACO blanks (1343-1390 nm on
V010506) drop out on their own. The two products must share a grid - they do,
since both are resampled from the same L1C - and a shape mismatch is reported
and the scene skipped.

Run inside the image, which has rasterio:

  docker run --rm -v "<with ac>:/paco:ro" -v "<without ac>:/ours:ro" \\
      -v "<out dir>:/out" --entrypoint python enmap-isofit:<tag> \\
      -m validate_vs_paco --paco /paco --ours /ours --csv /out/validation.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio

log = logging.getLogger("validate_vs_paco")

CLASSES = {"land": 1, "water": 2}
TABLE_BANDS = [0, 8, 16, 24, 32, 48, 64, 96, 112, 144, 176, 208, 223]
GRID_NM = (460.0, 2450.0)
MIN_PIXELS = 500


def find_pairs(paco_root: Path, ours_root: Path) -> list[tuple[str, Path, Path]]:
    """Match PACO and our SPECTRAL_IMAGE.TIF files by datatake id.

    Only `____L2A-` names are considered on our side, so L1C inputs and
    products from pre-1.1.3 runs (which carried the L1C id) are never picked.
    """
    def by_dt(root: Path) -> dict[str, Path]:
        out = {}
        for f in sorted(root.rglob("ENMAP01-____L2A-DT*-SPECTRAL_IMAGE.TIF")):
            out.setdefault(re.search(r"(DT\d{10})", f.name).group(1), f)
        return out

    paco, ours = by_dt(paco_root), by_dt(ours_root)
    for dt in sorted(set(ours) - set(paco)):
        log.warning("%s: no PACO product, skipped", dt)
    return [(dt, paco[dt], ours[dt]) for dt in sorted(set(paco) & set(ours))]


def sidecar(spectral: Path, suffix: str) -> Path:
    return spectral.with_name(spectral.name.replace("SPECTRAL_IMAGE.TIF", suffix))


def read_reflectance(path: Path, step: int) -> np.ndarray:
    """int16 x10000 -> float reflectance, nodata -> NaN, (bands, rows, cols)."""
    with rasterio.open(path) as src:
        a = src.read(out_dtype="float32")[:, ::step, ::step]
        return np.where(a == src.nodata, np.nan, a / 1e4)


def paco_wavelengths(paco_spectral: Path, n: int) -> np.ndarray:
    root = ET.parse(sidecar(paco_spectral, "METADATA.XML")).getroot()
    wl = [float(e.text) for e in root.iter("wavelengthCenterOfBand")]
    return np.array(wl[:n])


def spectral_angle_deg(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Per-pixel angle between two (bands, pixels) spectra; NaN bands as 0."""
    p, o = np.nan_to_num(p), np.nan_to_num(o)
    cos = (p * o).sum(0) / (np.linalg.norm(p, axis=0) * np.linalg.norm(o, axis=0) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def compare_class(dt, name, P, O, wl) -> tuple[list[dict], dict]:
    """P, O: (bands, pixels) for one class. Returns per-band rows and a summary."""
    ok = np.isfinite(P) & np.isfinite(O)
    bands = []
    for i in TABLE_BANDS:
        k = ok[i]
        if k.sum() < 100:
            continue
        p, o = P[i, k], O[i, k]
        bands.append(dict(scene=dt, cls=name, band=i + 1, nm=round(wl[i], 1),
                          paco=np.median(p), ours=np.median(o),
                          bias=np.median(o - p), r=np.corrcoef(p, o)[0, 1]))
    d = np.where(ok, O - P, np.nan)
    grid = (wl >= GRID_NM[0]) & (wl <= GRID_NM[1])
    covered = np.isfinite(P).mean(1) > 0.5
    summary = dict(
        scene=dt, cls=name, pixels=P.shape[1],
        mae_all=np.nanmean(np.abs(d)),
        mae_460_2450=np.nanmean(np.abs(d[grid])),
        mae_below_700=np.nanmean(np.abs(d[wl < 700])),
        mae_above_800=np.nanmean(np.abs(d[wl > 800])),
        bias_460=np.nanmedian(d[np.argmin(np.abs(wl - 460))]),
        sam_deg=np.median(spectral_angle_deg(P[covered], O[covered])),
    )
    return bands, summary


def compare_scene(dt: str, paco: Path, ours: Path, step: int):
    p, o = read_reflectance(paco, step), read_reflectance(ours, step)
    if p.shape != o.shape:
        log.warning("%s: shape mismatch PACO %s vs ours %s, skipped", dt, p.shape, o.shape)
        return [], []
    wl = paco_wavelengths(paco, p.shape[0])
    with rasterio.open(sidecar(paco, "QL_QUALITY_CLASSES.TIF")) as src:
        cls = src.read(1)[::step, ::step]
    bands, summaries = [], []
    for name, value in CLASSES.items():
        m = cls == value
        if m.sum() < MIN_PIXELS:
            continue
        b, s = compare_class(dt, name, p[:, m], o[:, m], wl)
        bands += b
        summaries.append(s)
    return bands, summaries


def print_tables(bands: list[dict], summaries: list[dict]) -> None:
    for s in summaries:
        print(f"\n== {s['scene']} {s['cls']} ({s['pixels']} px)")
        print("   band      nm    PACO    ours    bias      r")
        for b in bands:
            if (b["scene"], b["cls"]) == (s["scene"], s["cls"]):
                print(f"   {b['band']:4d} {b['nm']:7.1f}  {b['paco']:+.3f}  {b['ours']:+.3f}  {b['bias']:+.3f}  {b['r']:.3f}")
    print("\nscene         class  MAE_all  MAE_460-2450  MAE<700  MAE>800  bias@460  SAM_deg")
    for s in summaries:
        print(f"{s['scene']}  {s['cls']:5s}  {s['mae_all']:.4f}   {s['mae_460_2450']:.4f}        "
              f"{s['mae_below_700']:.4f}   {s['mae_above_800']:.4f}   {s['bias_460']:+.3f}    {s['sam_deg']:.2f}")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    log.info("Wrote %s", path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paco", required=True, help="Root holding PACO L2A product dirs")
    p.add_argument("--ours", required=True, help="Root holding this pipeline's L2A output")
    p.add_argument("--step", type=int, default=1, help="Use every Nth pixel each way (default 1 = all)")
    p.add_argument("--csv", help="Write the summary here; per-band rows go to <name>_bands.csv")
    args = p.parse_args(argv)

    pairs = find_pairs(Path(args.paco), Path(args.ours))
    if not pairs:
        log.error("No scene found in both roots")
        return 1
    bands, summaries = [], []
    for dt, paco, ours in pairs:
        log.info("Comparing %s", dt)
        b, s = compare_scene(dt, paco, ours, args.step)
        bands += b
        summaries += s
    print_tables(bands, summaries)
    if args.csv and summaries:
        out = Path(args.csv)
        write_csv(summaries, out)
        write_csv(bands, out.with_name(out.stem + "_bands.csv"))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
