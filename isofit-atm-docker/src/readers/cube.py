"""Generic cube access for ENVI (.hdr + .dat/.img/.bin/no extension) and GeoTIFF, whatever the sensor.

Both return a CubeInfo: shape, delivered interleave, wavelengths (nm, ascending), FWHM if delivered, a row reader
that yields float32 (rows, bands, samples) with every invalid value as NaN, and the georeference (rasterio for both
formats, so ENVI map info with rotation and GeoTIFF transforms are handled by the same, tested code).

Interleave is read from the ENVI header or the GeoTIFF's INTERLEAVE tag, then checked against the data itself
(check_interleave): a BIL/BSQ mix-up reads back as plausible-looking garbage rather than an error, so the reader
compares neighbouring-band correlation under each candidate layout and refuses a header that disagrees.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}
DATA_EXTS = ("", ".dat", ".img", ".bin", ".raw", ".bsq", ".bil", ".bip")
WL_IN_DESC = re.compile(r"\(?\s*([0-9]+(?:\.[0-9]+)?)\s*(nm|um|µm|micrometers?|nanometers?)\s*\)?", re.I)


@dataclass
class CubeInfo:
    path: str
    fmt: str
    shape: tuple[int, int, int]               # lines, bands, samples
    interleave: str
    wl: np.ndarray
    fwhm: np.ndarray | None
    read_rows: Callable[[int, int], np.ndarray]
    dtype: str
    meta: dict = field(default_factory=dict)  # header / tags, lower-case keys
    band_meta: list = field(default_factory=list)
    transform: object = None                  # rasterio Affine, None if not georeferenced
    crs: object = None
    notes: list = field(default_factory=list)


def _wl_nm(values, units: str | None) -> np.ndarray:
    wl = np.asarray(values, float)
    u = (units or "").strip().lower()
    if u.startswith(("um", "µm", "micro")) or (not u and np.nanmax(wl) < 100):
        wl = wl * 1000.0
    return wl


def load_wl_file(path: str, nb: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Operator-supplied spectral axis (--wavelength-file): one column (centres), two (centres, FWHM) or ISOFIT's
    three (index, centres, FWHM); nm or um. Must list exactly one row per band."""
    q = np.atleast_2d(np.loadtxt(path, ndmin=2))
    if q.shape[0] != nb:
        raise ValueError(f"{path}: {q.shape[0]} rows for {nb} bands")
    wl, fw = (q[:, 0], None) if q.shape[1] == 1 else (q[:, 0], q[:, 1]) if q.shape[1] == 2 else (q[:, 1], q[:, 2])
    nm = _wl_nm(wl, None)
    return nm, (fw * (nm[0] / wl[0]) if fw is not None else None)


def fix_order(wl: np.ndarray, notes: list) -> np.ndarray:
    """Repair isolated typos (Tanager ENVI exports carry e.g. 376.44 between 381 and 386): a band out of order
    while its neighbours are consistent gets their midpoint, noted. Returns the repaired copy."""
    wl = wl.copy()
    for i in range(1, len(wl) - 1):
        lo, hi = wl[i - 1], wl[i + 1]
        if lo < hi and not (lo < wl[i] < hi):
            notes.append(f"band {i + 1} wavelength {wl[i]:.2f} nm out of order; neighbour midpoint {(lo + hi) / 2:.2f} used")
            wl[i] = (lo + hi) / 2
    return wl


def _corr_rows(a: np.ndarray, b: np.ndarray) -> float:
    """Median over rows of the Pearson correlation between a[i] and b[i] (NaN-safe)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    n = ok.sum(1)
    keep = n > 10
    if not keep.any():
        return float("nan")
    a, b, ok, n = a[keep], b[keep], ok[keep], n[keep]
    ma, mb = a.sum(1) / n, b.sum(1) / n
    da, db = np.where(ok, a - ma[:, None], 0), np.where(ok, b - mb[:, None], 0)
    r = (da * db).sum(1) / np.sqrt((da * da).sum(1) * (db * db).sum(1) + 1e-30)
    return float(np.median(r))


def coherence(x: np.ndarray) -> float:
    """Median correlation of each pixel's spectrum with the block's mean spectrum. Real spectra share the sun's and
    the atmosphere's shape (radiance 0.99, reflectance 0.89 on AVIRIS-NG); a wrong layout makes each "spectrum" a
    slice of an image row, which does not (0.15-0.53)."""
    sp = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    sp = sp[np.all(np.isfinite(sp), 1)][::3]
    if len(sp) < 20:
        return float("nan")
    m = sp.mean(0)
    a, b = sp - sp.mean(1, keepdims=True), m - m.mean()
    return float(np.median((a @ b) / np.sqrt((a * a).sum(1) * (b * b).sum() + 1e-30)))


def layout_score(x: np.ndarray) -> float:
    """x: (rows, bands, samples) read under some layout. Two tests, multiplied:
    - coherence(): spectra share one shape only under the right layout;
    - smoothness: real imagery is smooth along bands, samples and lines alike, and each wrong layout breaks one
      (BSQ read as BIL turns neighbouring "bands" into lines of one band: 0.44 against 0.90 on AVIRIS-NG radiance,
      where coherence alone was 0.95 against 0.99). The weakest of the three axes counts.
    Tested on AVIRIS-NG radiance and reflectance written in all three layouts under all three headers."""
    x = x.astype(np.float64)
    return coherence(x) * _smoothness(x)


def _smoothness(x: np.ndarray) -> float:
    nb = x.shape[1]
    bands = np.arange(0, nb - 1, max(1, nb // 40))
    by_band = _corr_rows(x[:, bands, :].transpose(1, 0, 2).reshape(len(bands), -1),
                         x[:, bands + 1, :].transpose(1, 0, 2).reshape(len(bands), -1))
    by_sample = _corr_rows(x[:, bands, :-1].transpose(1, 0, 2).reshape(len(bands), -1),
                           x[:, bands, 1:].transpose(1, 0, 2).reshape(len(bands), -1))
    by_line = _corr_rows(x[:-1, bands, :].transpose(1, 0, 2).reshape(len(bands), -1),
                         x[1:, bands, :].transpose(1, 0, 2).reshape(len(bands), -1))
    return float(np.nanmin([by_band, by_sample, by_line]))


def check_interleave(sample: Callable[[str], np.ndarray], declared: str, notes: list) -> str:
    """Pick the layout that reads as real imagery (layout_score). sample(layout) returns a few consecutive lines,
    (rows, bands, samples), read under that layout. Keeps the declared layout unless another is clearly better."""
    score = {}
    for lay in ("bil", "bip", "bsq"):
        try:
            s = layout_score(sample(lay))
        except Exception:   # noqa: BLE001 - a layout that cannot even be read scores worst
            continue
        if np.isfinite(s):
            score[lay] = s
    if not score:
        return declared
    best = max(score, key=score.get)
    if declared in score and score[declared] >= score[best] - 0.05:
        return declared
    notes.append(f"header says interleave {declared!r} but the data read as {best!r} "
                 f"(layout score {score.get(declared, float('nan')):.2f} vs {score[best]:.2f}); using {best!r}")
    return best


def _georef(data_path: str):
    """rasterio's view of the file: (transform, crs) or (None, None) when it carries no georeference."""
    try:
        import rasterio
        with rasterio.open(data_path) as ds:
            t, crs = ds.transform, ds.crs
            if crs is None or t is None or t.is_identity:
                return None, None
            return t, crs
    except Exception:   # noqa: BLE001 - not readable by GDAL: no georeference
        return None, None


def open_envi(hdr: str, wl_file: str | None = None) -> CubeInfo:
    from spectral.io import envi
    meta = {k.lower(): v for k, v in envi.read_envi_header(hdr).items()}
    stem = hdr[:-4] if hdr.lower().endswith(".hdr") else hdr
    data = next((stem + e for e in DATA_EXTS if os.path.isfile(stem + e) and not (stem + e).lower().endswith(".hdr")), None)
    if data is None:
        data = next((hdr[:-4] + e for e in (".dat", ".img", ".bin") if os.path.isfile(hdr[:-4] + e)), None)
    if data is None:
        raise FileNotFoundError(f"no data file next to {hdr}")
    L, B, S = int(meta["lines"]), int(meta["bands"]), int(meta["samples"])
    dt = ENVI_DTYPES.get(int(meta.get("data type", 4)))
    if dt is None:
        raise ValueError(f"unsupported ENVI data type {meta.get('data type')}")
    dt = (">" if str(meta.get("byte order", "0")).strip() == "1" else "<") + dt
    off = int(meta.get("header offset", 0) or 0)
    need = off + L * B * S * np.dtype(dt).itemsize
    if os.path.getsize(data) < need:
        raise ValueError(f"{data} is {os.path.getsize(data)} bytes, header implies {need}")
    notes = []
    declared = str(meta.get("interleave", "bsq")).strip().lower()
    shapes = {"bsq": (B, L, S), "bil": (L, B, S), "bip": (L, S, B)}
    mm = {lay: np.memmap(data, dtype=dt, mode="r", offset=off, shape=shp) for lay, shp in shapes.items()}

    def rows_as(lay, r0, r1):
        m = mm[lay]
        blk = m[:, r0:r1, :].transpose(1, 0, 2) if lay == "bsq" else m[r0:r1] if lay == "bil" else m[r0:r1].transpose(0, 2, 1)
        return np.array(blk, dtype=np.float32, copy=True)   # a float32 memmap slice would be a read-only view

    ignore = meta.get("data ignore value")
    ignore = float(ignore) if ignore not in (None, "") else None

    def probe(lay):
        mid = max(0, L // 2 - 4)
        x = rows_as(lay, mid, min(L, mid + 8))
        return np.where(x == np.float32(ignore), np.nan, x) if ignore is not None else x
    inter = check_interleave(probe, declared, notes)

    wl = _wl_nm(meta["wavelength"], meta.get("wavelength units")) if "wavelength" in meta else None
    fwhm = np.asarray(meta["fwhm"], float)[:B] if "fwhm" in meta and len(meta["fwhm"]) >= B else None
    if fwhm is not None and wl is not None and wl.max() > 100 and np.nanmax(fwhm) < 1:
        fwhm = fwhm * 1000.0
    if wl_file:
        wl, f2 = load_wl_file(wl_file, B)
        fwhm = f2 if f2 is not None else fwhm
        notes.append(f"wavelengths from the operator's file {wl_file}")
    if wl is None or len(wl) != B:
        raise ValueError(f"{hdr}: no wavelength list for its {B} bands; the spectral axis is required "
                         "(pass --wavelength-file)")
    scale = meta.get("reflectance scale factor")

    def read_rows(r0, r1):
        x = rows_as(inter, r0, r1)
        bad = ~np.isfinite(x)
        if ignore is not None:
            bad |= x == np.float32(ignore)
        x[bad] = np.nan
        return x

    t, crs = _georef(data)
    if scale:
        meta["_scale"] = float(scale)
    return CubeInfo(data, "envi", (L, B, S), inter, wl, fwhm, read_rows, dt, meta, [], t, crs, notes)


def open_geotiff(path: str, wl_file: str | None = None) -> CubeInfo:
    import rasterio
    ds = rasterio.open(path)
    L, B, S = ds.height, ds.count, ds.width
    meta = {k.lower(): v for k, v in ds.tags().items()}
    for dom in ("ENVI", "IMAGE_STRUCTURE"):
        try:
            meta.update({f"{dom.lower()}:{k.lower()}": v for k, v in ds.tags(ns=dom).items()})
        except Exception:   # noqa: BLE001
            pass
    bmeta = [{k.lower(): v for k, v in ds.tags(i).items()} for i in range(1, B + 1)]
    notes, wl = [], []
    for i, (bm, desc) in enumerate(zip(bmeta, ds.descriptions)):
        v = bm.get("wavelength") or bm.get("wavelength_nm") or bm.get("center_wavelength")
        if v is None and desc:
            m = WL_IN_DESC.search(desc)
            v = m.group(1) if m else None
        wl.append(float(v) if v is not None else np.nan)
    wl = np.asarray(wl)
    if np.isnan(wl).any() and "envi:wavelength" in meta:
        wl = np.asarray([float(x) for x in meta["envi:wavelength"].strip("{}").split(",")])
    sidecar_fwhm = None
    if wl_file:
        wl, sidecar_fwhm = load_wl_file(wl_file, B)
        notes.append(f"wavelengths from the operator's file {wl_file}")
    elif np.isnan(wl).any() or len(wl) != B:   # EnMAP COGs keep the spectral axis in the sibling METADATA.XML
        wl, sidecar_fwhm = _enmap_xml_axis(path, B)
        if wl is None:
            raise ValueError(f"{path}: no per-band wavelength (band tags, descriptions, ENVI metadata or an EnMAP "
                             "METADATA.XML next to it); required (pass --wavelength-file)")
        notes.append("wavelengths and FWHM from the EnMAP METADATA.XML next to the image")
    units = (bmeta[0].get("wavelength_units") or bmeta[0].get("wavelenght_units")   # GHOSt's spelling
             or meta.get("wavelength_units") or meta.get("envi:wavelength_units"))
    wl = _wl_nm(wl, units)
    fw = [bm.get("fwhm") for bm in bmeta]
    fwhm = np.asarray(fw, float) if all(f is not None for f in fw) else sidecar_fwhm
    nodata = ds.nodatavals[0]
    scales, offsets = np.asarray(ds.scales, float), np.asarray(ds.offsets, float)
    rescale = not (np.allclose(scales, 1) and np.allclose(offsets, 0))

    def read_rows(r0, r1):
        from rasterio.windows import Window
        x = ds.read(window=Window(0, r0, S, r1 - r0)).astype(np.float32)   # (bands, rows, samples)
        bad = ~np.isfinite(x)
        if nodata is not None and np.isfinite(nodata):
            bad |= x == np.float32(nodata)
        if rescale:
            x = x * scales[:, None, None].astype(np.float32) + offsets[:, None, None].astype(np.float32)
        x[bad] = np.nan
        return np.ascontiguousarray(x.transpose(1, 0, 2))

    inter = str(meta.get("image_structure:interleave", "band")).lower()
    inter = {"band": "bsq", "pixel": "bip", "line": "bil"}.get(inter, inter)
    t, crs = (ds.transform, ds.crs) if ds.crs is not None and not ds.transform.is_identity else (None, None)
    if rescale:
        notes.append("GDAL scale/offset applied")
    return CubeInfo(path, "geotiff", (L, B, S), inter, wl, fwhm, read_rows, ds.dtypes[0], meta, bmeta, t, crs, notes)


def _enmap_xml_axis(path: str, nb: int):
    """(wavelengths, fwhm) from an EnMAP METADATA.XML in the image's folder, or (None, None)."""
    xml = sorted(Path(path).parent.glob("*METADATA.XML"))
    if not xml:
        return None, None
    try:
        import enmap_l1c_preprocess as E
        bands = E.parse_metadata(xml[0])["bands"]
    except Exception:   # noqa: BLE001 - not an EnMAP XML we can read
        return None, None
    if len(bands) != nb:
        return None, None
    return np.array([b["wl"] for b in bands], float), np.array([b["fwhm"] for b in bands], float)


def sort_bands(c: CubeInfo) -> CubeInfo:
    """Make wavelengths ascending (descending deliveries, isolated typos), permuting the reader to match."""
    wl = fix_order(c.wl, c.notes)
    order = np.argsort(wl, kind="stable")
    if np.any(np.diff(wl[order]) <= 0):
        dup = int(np.sum(np.diff(wl[order]) <= 0))
        c.notes.append(f"{dup} duplicate band centres kept (overlapping detectors)")
    if np.array_equal(order, np.arange(len(wl))):
        c.wl = wl
        return c
    base = c.read_rows
    c.read_rows = lambda r0, r1: base(r0, r1)[:, order, :]
    c.wl, c.fwhm = wl[order], (c.fwhm[order] if c.fwhm is not None else None)
    c.notes.append("bands reordered to ascending wavelength")
    return c


def derive_fwhm(wl: np.ndarray) -> np.ndarray:
    """FWHM = the smaller neighbour spacing (x1.0): uniform axes give the spacing; at the edge of a gap the true
    sampling is kept instead of averaging across the hole (GHOSt dataset 3 drops 25 bands)."""
    d = np.diff(wl)
    left, right = np.r_[d[0], d], np.r_[d, d[-1]]
    return np.abs(np.minimum(left, right))


def lonlat_reader(t, crs) -> Callable | None:
    """(r0, r1) -> (lon, lat) per pixel centre, for a georeferenced cube."""
    if t is None or crs is None:
        return None
    from pyproj import Transformer
    tr = None if crs.to_epsg() == 4326 else Transformer.from_crs(crs, 4326, always_xy=True)

    def f(r0, r1, ns):
        cc, rr = np.meshgrid(np.arange(ns) + 0.5, np.arange(r0, r1) + 0.5)
        x, y = t.a * cc + t.b * rr + t.c, t.d * cc + t.e * rr + t.f
        if tr is not None:
            x, y = tr.transform(x, y)
        return np.asarray(x, np.float64), np.asarray(y, np.float64)
    return f


def find_hdr(path: str) -> str | None:
    p = Path(path)
    for c in (p.with_suffix(".hdr"), Path(str(p) + ".hdr")):
        if c.is_file():
            return str(c)
    return str(p) if p.suffix.lower() == ".hdr" and p.is_file() else None
