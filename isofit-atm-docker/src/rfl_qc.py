"""
Quality masks on ISOFIT's float ENVI reflectance, applied in place before repacking.

Both steps write nodata (-9999) and never alter a value in range:

- mask_windows: ISOFIT fits only inside its inversion windows. Bands outside come
  back near 0 and, left in, pull the 1960/1970 nm Allotrope grid bands to ~0 (PACO
  has 0.17 there). As nodata, the band filter interpolates them from valid
  neighbours, as it does for PACO's own blanks.
- clip_outliers: ISOFIT occasionally returns wild values (-8971, +514 on
  Himalayan snow and shadow). Negative ones become nodata. Values above the range
  are kept and flagged in <rfl>_outlier, because real flares and fires also read
  above 2.5, and they are what the anomaly stage is for. The test covers only
  the 165-band grid range outside the absorption windows; sensor-edge and
  absorption bands are noisy and, tested, removed 35-80% of AVIRIS-NG pixels.

A third step only reads: snow_mask writes a separate uint8 mask (<rfl>_snow, ENVI)
from NDSI on our own reflectance. JPL ships no snow mask for AVIRIS-NG, and
harmonisation leaves snow untouched, so every sensor needs one at deployment. On
the training scenes it matched DLR's and ASI's own snow flags with precision
0.91-1.00 and recall 0.92-1.00 (AC26-SN). GREEN_MIN 0.20 keeps lowland false alarms
(surf, river edges, bright roofs) at or below 0.23%. Salt flats still read as snow,
as they do in DLR's flag.
"""
from __future__ import annotations

import numpy as np
from spectral import envi

NODATA = -9999.0
# apply_oe defaults; the bands it actually inverts.
INVERSION_WINDOWS_NM = [(350.0, 1360.0), (1410.0, 1800.0), (1970.0, 2500.0)]
# Ranges not tested for outliers: below/above the 460-2450 nm grid and the absorption windows.
UNTESTED_NM = [(0, 460), (912, 978), (1131, 1152), (1350, 1450), (1800, 1950), (2450, 3000)]
VALID_RANGE = (-0.2, 2.5)
# NDSI snow rule (AC26-SN): NDSI(560, 1600) > NDSI_MIN, NIR > NIR_MIN (excludes water), green > GREEN_MIN.
NDSI_MIN, NIR_MIN, GREEN_MIN = 0.4, 0.11, 0.20
SNOW, CLEAR, MASK_NODATA = 1, 0, 255
OUT_HIGH, OUT_LOW = 1, 2   # clip_outliers mask values


def _open(rfl: str):
    """Writable (lines, bands, samples) memmap of an ISOFIT BIL cube, and its wavelengths in nm."""
    img = envi.open(rfl + ".hdr", rfl)
    wl = np.array(img.metadata["wavelength"], float)
    wl = wl * 1000 if wl.max() < 100 else wl
    return img.open_memmap(interleave="bil", writable=True), wl


def _in_any(wl: np.ndarray, ranges) -> np.ndarray:
    return np.any([(wl >= lo) & (wl <= hi) for lo, hi in ranges], axis=0)


def mask_windows(rfl: str) -> int:
    """Set bands outside ISOFIT's inversion windows to nodata. Returns the band count masked."""
    m, wl = _open(rfl)
    out = ~_in_any(wl, INVERSION_WINDOWS_NM)
    m[:, out, :] = NODATA
    m.flush()
    return int(out.sum())


def clip_outliers(rfl: str) -> tuple[int, int, str]:
    """Test the tested bands against VALID_RANGE and write <rfl>_outlier (uint8, ENVI).

    Below the range (negative reflectance, a failed inversion): the pixel becomes nodata in every band, as before.
    Above it: the values are kept and the pixel is flagged. Gas flares, fires and hot metal are what the
    anomaly stage looks for, and nodata would erase them. ISOFIT's rare wild positives (+514 on Himalayan snow)
    are kept too, flagged the same way.
    Mask: CLEAR, OUT_HIGH kept and flagged, OUT_LOW set to nodata, MASK_NODATA nodata in every tested band.
    Returns (pixels set to nodata, pixels flagged high, mask path).
    """
    m, wl = _open(rfl)
    low = np.zeros((m.shape[0], m.shape[2]), bool)
    high, empty = np.zeros_like(low), np.ones_like(low)
    for b in np.flatnonzero(~_in_any(wl, UNTESTED_NM)):
        x = m[:, b, :]
        valid = x > NODATA + 9
        low |= valid & (x < VALID_RANGE[0])
        high |= valid & (x > VALID_RANGE[1])
        empty &= ~valid
    high &= ~low
    for b in range(m.shape[1]):
        m[:, b, :][low] = NODATA
    m.flush()
    mask = np.select([low, high, empty], [OUT_LOW, OUT_HIGH, MASK_NODATA], CLEAR).astype(np.uint8)
    path = rfl + "_outlier"
    envi.save_image(path + ".hdr", mask[..., None], dtype=np.uint8, interleave="bsq", force=True, ext="",
                    metadata={"description": "reflectance range flags (rfl_qc.clip_outliers): 1 above 2.5, kept; "
                                             "2 below -0.2, set to nodata", "data ignore value": MASK_NODATA,
                              "band names": ["outlier"]})
    return int(low.sum()), int(high.sum()), path


def snow_mask(rfl: str) -> tuple[str, float]:
    """Write <rfl>_snow (uint8: 1 snow, 0 clear, 255 nodata). Returns its path and the snow fraction of valid pixels."""
    m, wl = _open(rfl)
    band = lambda nm: np.asarray(m[:, int(np.argmin(np.abs(wl - nm))), :], np.float32)
    g, s, n = band(560), band(1600), band(860)
    valid = (g > NODATA + 9) & (s > NODATA + 9) & (n > NODATA + 9)
    snow = valid & ((g - s) / (g + s + 1e-6) > NDSI_MIN) & (n > NIR_MIN) & (g > GREEN_MIN)
    out = np.where(valid, np.where(snow, SNOW, CLEAR), MASK_NODATA).astype(np.uint8)[..., None]
    path = rfl + "_snow"
    envi.save_image(path + ".hdr", out, dtype=np.uint8, interleave="bsq", force=True, ext="",   # ISOFIT naming: no .img
                    metadata={"description": "NDSI snow mask (rfl_qc.snow_mask)", "data ignore value": MASK_NODATA,
                              "band names": ["snow"]})
    return path, float(snow.sum() / max(valid.sum(), 1))
