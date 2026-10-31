"""What level is this cube: radiance, TOA reflectance, surface reflectance, or raw counts?

The one mistake this module must not make is to call a TOA product surface reflectance: that product would pass
through uncorrected and be submitted as L2. So the rule is asymmetric. A cube passes through as surface reflectance
only when at least MIN_SR_TESTS of the atmosphere tests could run and every one of them votes SR with margin; any
TOA vote, too few tests, or a value inside a test's grey zone means "correct it" (toa_reflectance). Calling an SR
product TOA costs a needless correction and is flagged; the reverse would be silent.

Tests, all on a pixel sample, scale-free where possible:
  shape      radiance carries the sun's spectrum: L(2200)/L(550) is under 0.3 for every surface (snow ~0.003,
             soil ~0.1), while reflectance reaches 0.5+ on the soil / rock / vegetation found in almost any scene.
             Uses the 90th percentile of the per-pixel ratio. Separates radiance from reflectance.
  o2_760     depth of the oxygen A band (760 vs 740/780 nm): TOA 0.16-0.59 (shallowest at PRISMA's 10-15 nm FWHM);
             vendor SR <= 0.025. TOA above 0.12, SR below 0.06.
  h2o_940    depth of the 940 nm water band (vs 865/1040): TOA 0.57-0.93; vendor SR up to 0.20 (ASI L2D keeps
             residual water). TOA above 0.40, SR below 0.30.
  h2o_1135   the same at 1135 nm (vs 1040/1240).
  co2_2010   depth of the 2010 nm CO2 band (1960 / 2080 nm continuum): TOA 0.7-0.9 with this continuum. Votes TOA
             only: on SR products the 1960 nm point is in the water band's noisy wing (one JPL flight read 0.26).
  sat_1380   1380 nm over its continuum (1250): saturated water absorption, TOA 0.002-0.013 (< 0.15 votes TOA);
             a negative median votes SR (only a processed product goes negative); anything else abstains, since
             vendor SR fills the band with noise or interpolated continuum (0.23-175 here).
  blue_floor TOA reflectance never gets darker than the Rayleigh path: p1 at 450 nm above 0.06 is TOA-like.
             Votes TOA only (a bright desert in SR also clears 0.06, and voting TOA there is the safe error).
Depths use the scene's median spectrum, so a test runs only when the bands exist and are valid in it.
"""
from __future__ import annotations

import numpy as np

MIN_SR_TESTS = 3
# (name, centre nm, continuum nm, TOA if depth >, SR if depth <)
# Set on the confusion table of 2026-10-31 (AC26-LVL: 9 radiance, 9 TOA, 10 vendor SR cubes from AVIRIS-NG, Tanager,
# EnMAP, PRISMA): TOA O2 depth 0.158 (PRISMA, 10-15 nm FWHM) to 0.59, SR <= 0.025; TOA water depths >= 0.568, vendor
# SR <= 0.20 (ASI L2D keeps residual water features). O2 scales with surface pressure, so it stays above 0.06 on a
# TOA cube up to about 7 km: it is the test that blocks a false SR. In-sample thresholds; GHOSt is unseen.
DEPTH_TESTS = (("o2_760", 761, (740, 780), 0.12, 0.06),
               ("h2o_940", 940, (865, 1040), 0.40, 0.30),
               ("h2o_1135", 1135, (1040, 1240), 0.40, 0.30),
               ("co2_2010", 2010, (1960, 2080), 0.50, -1.0))   # TOA-only (see the module docstring)


def _band(wl, nm, tol=12.0):
    i = int(np.argmin(np.abs(wl - nm)))
    return i if abs(wl[i] - nm) <= tol else None


def _depth(spec, wl, centre, cont):
    """1 - value at the band centre / continuum interpolated between cont[0] and cont[1] (min of 3 bands at the
    centre, so the depth does not depend on where the band grid samples the feature)."""
    ic, a, b = _band(wl, centre), _band(wl, cont[0]), _band(wl, cont[1])
    if None in (ic, a, b):
        return None
    lo, hi = max(0, ic - 1), min(len(wl), ic + 2)
    c = np.nanmin(spec[lo:hi])
    ca, cb = spec[a], spec[b]
    if not np.isfinite([c, ca, cb]).all() or ca <= 0 or cb <= 0:
        return None
    base = ca + (cb - ca) * (wl[ic] - wl[a]) / (wl[b] - wl[a])
    return float(1.0 - c / base)


def classify(sample: np.ndarray, wl: np.ndarray, integer: bool = False) -> dict:
    """sample: (pixels, bands) of valid-ish pixels in delivered units. Returns the verdict and every test."""
    px = sample[np.isfinite(sample).mean(1) > 0.8]
    out = {"pixels": int(len(px)), "tests": {}}
    if len(px) < 50:
        return out | {"level": "unknown", "reason": "fewer than 50 valid pixels"}
    t = out["tests"]
    # shape: radiance vs reflectance
    a, b = _band(wl, 550, 20), _band(wl, 2200, 40)
    if a is not None and b is not None:
        ok = (px[:, a] > 0) & np.isfinite(px[:, b])
        ratio = float(np.percentile(px[ok, b] / px[ok, a], 90)) if ok.sum() > 50 else None
        t["shape"] = {"value": ratio, "vote": None if ratio is None else
                      "radiance" if ratio < 0.3 else "reflectance" if ratio > 0.5 else None}
    kind = t.get("shape", {}).get("vote")
    med = np.nanmedian(px, 0)
    p99 = float(np.nanpercentile(px, 99))
    out["p99"] = p99
    if kind == "radiance":
        if integer:   # integers left unscaled by the reader: no gain/offset was delivered with them
            return out | {"level": "dn", "reason": "integer counts with the sun's spectral shape and no calibration"}
        return out | {"level": "radiance", "reason": "solar spectral shape"}
    if kind is None:
        return out | {"level": "unknown", "reason": "spectral shape is neither radiance-like nor reflectance-like"}
    # reflectance: scale (x1e4 integer products) then TOA vs SR
    scale = 1.0 if p99 < 5 else 1e-4 if p99 < 2e4 else None
    out["reflectance_scale"] = scale
    if scale is None:   # EnMAP L1C raw counts read this way (p99 ~48000) and show no absorption at the band centres
        return out | {"level": "unknown", "reason": f"reflectance-like shape but values up to {p99:.0f} fit no "
                                                    "reflectance scale (0-1 or x1e-4): uncalibrated counts?"}
    for name, c, cont, toa, sr in DEPTH_TESTS:
        d = _depth(med, wl, c, cont)
        if d is not None:
            t[name] = {"value": d, "vote": "toa" if d > toa else "sr" if d < sr else None}
    i1380, i1250 = _band(wl, 1380, 10), _band(wl, 1250, 15)
    if i1380 is not None and i1250 is not None and np.isfinite(med[i1380]) and med[i1250] > 0 and med[i1380] != 0:
        r = float(med[i1380] / med[i1250])
        # Negative median radiance / TOA reflectance cannot happen; a negative median is a processed product.
        t["sat_1380"] = {"value": r, "vote": "toa" if 0 < r < 0.15 else "sr" if r < 0 else None}
    i450 = _band(wl, 450, 15)
    if i450 is not None and scale:
        p1 = float(np.nanpercentile(px[:, i450], 1)) * scale
        t["blue_floor"] = {"value": p1, "vote": "toa" if p1 > 0.06 else None}
    votes = [v["vote"] for k, v in t.items() if k != "shape"]
    # SR tests: the three depth tests; 1380 counts only when it votes (vendor SR fills it with anything from noise to
    # interpolated continuum, 0.23-175 x the 1250 nm value). co2 and blue_floor vote TOA only.
    ran = [k for k in ("o2_760", "h2o_940", "h2o_1135") if k in t and t[k]["value"] is not None]
    ran += [k for k in ("sat_1380",) if k in t and t[k]["vote"] is not None]
    n_sr, n_toa = votes.count("sr"), votes.count("toa")
    out["sr_tests_run"], out["sr_votes"], out["toa_votes"] = len(ran), n_sr, n_toa
    if n_toa == 0 and len(ran) >= MIN_SR_TESTS and n_sr == len(ran):
        return out | {"level": "surface_reflectance", "reason": f"all {n_sr} atmosphere tests vote SR"}
    why = (f"{n_toa} test(s) vote TOA" if n_toa else
           f"only {len(ran)} atmosphere test(s) could run (need {MIN_SR_TESTS})" if len(ran) < MIN_SR_TESTS else
           f"{len(ran) - n_sr} test(s) in the grey zone")
    return out | {"level": "toa_reflectance", "reason": f"not proven SR: {why}; corrected as TOA"}


def sample_pixels(read_rows, lines: int, n_blocks: int = 12, rows: int = 4, step: int = 3) -> np.ndarray:
    """(pixels, bands) from n_blocks evenly spaced row blocks, every step-th sample."""
    out = []
    for r0 in np.linspace(0, max(0, lines - rows), n_blocks).astype(int):
        x = read_rows(int(r0), int(min(lines, r0 + rows)))[:, :, ::step].astype(np.float32)
        x[x <= -9990] = np.nan   # -9999 fill in a file whose header does not declare it
        out.append(x.transpose(0, 2, 1).reshape(-1, x.shape[1]))
    return np.concatenate(out)
