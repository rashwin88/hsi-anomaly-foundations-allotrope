"""
EnMAP L1C -> ISOFIT input preprocessor.

Reads a DLR EnMAP L1C product directory and emits the three ENVI cubes that
ISOFIT's `apply_oe` expects:

  * rdn : TOA radiance   (uW/cm^2/sr/nm, float32, BIL)
  * loc : lon/lat/elev   (float32, BIP)
  * obs : 10-band obs    (float32, BIP)

Schema-tolerant against multiple EnMAP L1C metadata versions (v00.05, v00.06,
v00.07 including the level_X root introduced in processor v01.05.06).

Falls back gracefully when angle fields are missing or empty:
  1. try the field in XML
  2. try alternate names (sceneSZA, acrossOffNadirAngle, etc.)
  3. compute solar geometry from acquisition time + scene center lat/lon
  4. use a documented near-nadir default for view geometry
"""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

log = logging.getLogger("enmap_l1c_preprocess")

# EnMAP L1C radiance is delivered in W / (m^2 * sr * nm) after applying
# per-band gain and offset (per DLR User Manual). ISOFIT expects
# uW / (cm^2 * sr * nm).
#   1 W/m^2/sr/nm = 100 uW/cm^2/sr/nm
RAD_UNIT_SCALE = 100.0

# Fallback if <altitudeCoverage> is missing from the XML.
DEFAULT_ORBIT_ALT_KM = 653.0

BAD_BAND_RANGES_NM = [(1355, 1450), (1795, 1970)]


# ---------------------------------------------------------------------------
# XML helpers (schema-tolerant)
# ---------------------------------------------------------------------------

def _strip_ns(root: ET.Element) -> None:
    for elem in root.iter():
        elem.tag = re.sub(r"^\{.*\}", "", elem.tag)


def _text_of(root: ET.Element, xpath: str) -> str | None:
    """Return stripped text of first xpath match, or None."""
    el = root.find(xpath)
    if el is not None and el.text is not None:
        t = el.text.strip()
        return t if t else None
    return None


def _text_of_any(root: ET.Element, tag_names) -> str | None:
    """Walk the whole tree, return stripped text of first element whose local
    tag matches any name in tag_names. Handles multi-line/whitespace text."""
    if isinstance(tag_names, str):
        tag_names = [tag_names]
    tag_set = set(tag_names)
    for el in root.iter():
        local = el.tag.rsplit("}", 1)[-1]
        if local in tag_set and el.text is not None:
            t = el.text.strip()
            if t:
                return t
    return None

def _angle_value(root: ET.Element, tag_name: str) -> float | None:
    """Extract a scene-mean angle from an EnMAP L1C angle container.

    v00.07 schema stores angles as containers with per-corner children:
        <sunElevationAngle>
            <upper_left unit="DEG">65.77</upper_left>
            <center unit="DEG">65.96</center>
            ...
        </sunElevationAngle>

    Returns <center> if present, else the mean of the 4 corners,
    else the element's own leaf text (for older schemas), else None.
    """
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] != tag_name:
            continue
        center = el.find("center")
        if center is not None and center.text and center.text.strip():
            try:
                return float(center.text.strip())
            except ValueError:
                pass
        corner_vals = []
        for corner in ("upper_left", "upper_right", "lower_left", "lower_right"):
            child = el.find(corner)
            if child is not None and child.text and child.text.strip():
                try:
                    corner_vals.append(float(child.text.strip()))
                except ValueError:
                    pass
        if corner_vals:
            return sum(corner_vals) / len(corner_vals)
        if el.text and el.text.strip():
            try:
                return float(el.text.strip())
            except ValueError:
                pass
    return None

def _find_center_point(root: ET.Element) -> tuple[float | None, float | None]:
    """Extract scene center lat/lon from boundingPolygon/point[frame='center']."""
    for pt in root.iter():
        if pt.tag.rsplit("}", 1)[-1] != "point":
            continue
        frame_el = pt.find("frame")
        if frame_el is None or frame_el.text is None:
            continue
        if frame_el.text.strip().lower() != "center":
            continue
        lat_el = pt.find("latitude")
        lon_el = pt.find("longitude")
        lat = float(lat_el.text) if lat_el is not None and lat_el.text else None
        lon = float(lon_el.text) if lon_el is not None and lon_el.text else None
        return lat, lon
    return None, None


def _compute_solar_geometry(dt_utc: datetime, lat: float, lon: float
                            ) -> tuple[float, float]:
    """Compute (solar_zenith_deg, solar_azimuth_deg) using standard NOAA
    astronomical formulas. Good to ~0.1 deg for our purposes."""
    day_of_year = dt_utc.timetuple().tm_yday
    hour_utc = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0

    # Fractional year (radians)
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1 + (hour_utc - 12) / 24.0)

    # Equation of time (minutes)
    eqtime = 229.18 * (0.000075
                       + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))

    # Solar declination (radians)
    decl = (0.006918
            - 0.399912 * math.cos(gamma)
            + 0.070257 * math.sin(gamma)
            - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma)
            - 0.002697 * math.cos(3 * gamma)
            + 0.00148 * math.sin(3 * gamma))

    # True solar time (minutes)
    time_offset = eqtime + 4.0 * lon
    tst = hour_utc * 60.0 + time_offset

    # Hour angle (degrees, then radians)
    ha = math.radians(tst / 4.0 - 180.0)

    lat_rad = math.radians(lat)

    cos_zenith = (math.sin(lat_rad) * math.sin(decl)
                  + math.cos(lat_rad) * math.cos(decl) * math.cos(ha))
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))

    # Solar azimuth (measured clockwise from north)
    cos_az = ((math.sin(decl) - math.sin(lat_rad) * math.cos(math.radians(zenith)))
              / (math.cos(lat_rad) * math.sin(math.radians(zenith))))
    cos_az = max(-1.0, min(1.0, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:
        azimuth = 360.0 - azimuth

    return zenith, azimuth


# ---------------------------------------------------------------------------
# METADATA.XML parsing
# ---------------------------------------------------------------------------

def parse_metadata(xml_path: Path) -> dict:
    """Parse EnMAP L1C METADATA.XML across schema versions.

    Uses schema-agnostic tag lookup so it works with the level_X root schema
    (v00.07 / processor 01.05.06+) as well as older nested variants.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    _strip_ns(root)

    meta: dict = {}

    # -- Scene ID: filename in <metadata><name>, strip -METADATA.XML suffix --
    name_full = (_text_of(root, ".//metadata/name")
                 or _text_of_any(root, ["name"]))
    if not name_full:
        raise KeyError("Could not find scene name in METADATA.XML")
    scene_id = name_full
    for suffix in ("-METADATA.XML", "_METADATA.XML", "-METADATA.xml"):
        if scene_id.endswith(suffix):
            scene_id = scene_id[:-len(suffix)]
            break
    meta["scene_id"] = scene_id

    # -- Acquisition time --
    t_str = (_text_of(root, ".//base/temporalCoverage/startTime")
             or _text_of(root, ".//temporalCoverage/startTime")
             or _text_of_any(root, ["startTime"]))
    if not t_str:
        raise KeyError("Could not find startTime in METADATA.XML")
    meta["acquisition_time"] = datetime.fromisoformat(t_str.replace("Z", "+00:00"))

    # -- Scene center from boundingPolygon --
    center_lat, center_lon = _find_center_point(root)
    meta["center_lat"] = center_lat
    meta["center_lon"] = center_lon

    # -- Orbit altitude (meters -> km) --
    alt_m = _text_of_any(root, ["altitudeCoverage"])
    meta["orbit_altitude_km"] = (float(alt_m) / 1000.0 if alt_m
                                  else DEFAULT_ORBIT_ALT_KM)

        # -- Solar geometry: cascade of fallbacks -------------------------------
    sun_elev = _angle_value(root, "sunElevationAngle")
    sun_az = _angle_value(root, "sunAzimuthAngle")

    if sun_elev is not None:
        meta["solar_zenith"] = 90.0 - sun_elev
        meta["_solar_source"] = "sunElevationAngle/center"
    else:
        scene_sza = _text_of_any(root, ["sceneSZA"])
        if scene_sza:
            meta["solar_zenith"] = float(scene_sza)
            meta["_solar_source"] = "sceneSZA (integer fallback)"
        elif center_lat is not None and center_lon is not None:
            sza, saa = _compute_solar_geometry(
                meta["acquisition_time"], center_lat, center_lon)
            meta["solar_zenith"] = sza
            if sun_az is None:
                sun_az = saa
            meta["_solar_source"] = "computed"
        else:
            raise KeyError("No solar geometry and cannot compute (missing center lat/lon)")

    if sun_az is not None:
        meta["solar_azimuth"] = float(sun_az)
    elif center_lat is not None and center_lon is not None:
        _, saa = _compute_solar_geometry(
            meta["acquisition_time"], center_lat, center_lon)
        meta["solar_azimuth"] = saa
    else:
        log.warning("No solar_azimuth; using fallback 180.0")
        meta["solar_azimuth"] = 180.0

    # -- View geometry: EnMAP is a near-nadir pushbroom ---------------------
    vza = _angle_value(root, "viewingZenithAngle")
    vaa = _angle_value(root, "viewingAzimuthAngle")
    across = _angle_value(root, "acrossOffNadirAngle")
    scene_az = _angle_value(root, "sceneAzimuthAngle")

    if vza is not None:
        meta["view_zenith"] = vza
    elif across is not None:
        meta["view_zenith"] = abs(across)
        log.info(f"Using acrossOffNadirAngle={across:.3f} for view_zenith")
    else:
        log.warning("No view_zenith in XML; using 0.0 (near-nadir assumption)")
        meta["view_zenith"] = 0.0

    if vaa is not None:
        meta["view_azimuth"] = vaa
    elif scene_az is not None:
        meta["view_azimuth"] = scene_az
    else:
        log.warning("No view_azimuth in XML; using fallback 180.0")
        meta["view_azimuth"] = 180.0
      
    # -- Bands: bandCharacterisation/bandID (stable across schemas) ---------
    bands: list[dict] = []
    for band in root.iter():
        if band.tag.rsplit("}", 1)[-1] != "bandID":
            continue    
        # Skip bandID references in dead-pixel maps / quality flags — they
        # exist elsewhere in the XML without wavelength info.
        if band.findtext("wavelengthCenterOfBand") is None:
            continue    
        try:
            idx = int(band.attrib.get("number", band.attrib.get("id", 0)))
            wl = float(band.findtext("wavelengthCenterOfBand"))
            fwhm = float(band.findtext("FWHMOfBand"))
            gain = float(band.findtext("GainOfBand"))
            offset_str = band.findtext("OffsetOfBand")
            offset = float(offset_str) if offset_str else 0.0
        except (TypeError, ValueError) as e:
            log.warning(f"Skipping malformed bandID: {e}")
            continue
        bands.append({"idx": idx, "wl": wl, "fwhm": fwhm,
                      "gain": gain, "offset": offset})
    if not bands:
        raise RuntimeError("No bandID elements found in METADATA.XML")
    bands.sort(key=lambda b: b["idx"])
    meta["bands"] = bands

    log.info(
        "Scene: %s | %d bands | SZA=%.2f (%s) SAA=%.2f VZA=%.2f VAA=%.2f "
        "| center=(%.3f, %.3f) | orbit=%.1f km",
        meta["scene_id"], len(bands),
        meta["solar_zenith"], meta.get("_solar_source", "?"),
        meta["solar_azimuth"], meta["view_zenith"], meta["view_azimuth"],
        center_lat or 0.0, center_lon or 0.0, meta["orbit_altitude_km"],
    )
    return meta


def find_scene_files(input_dir: Path) -> tuple[Path, Path]:
    """Locate the spectral image and metadata XML in an EnMAP L1C directory."""
    xml_candidates = (list(input_dir.glob("*METADATA.XML"))
                      + list(input_dir.glob("*metadata.xml"))
                      + list(input_dir.glob("*-METADATA.XML")))
    if not xml_candidates:
        raise FileNotFoundError(f"No METADATA.XML found in {input_dir}")
    xml_path = xml_candidates[0]

    img_candidates = (list(input_dir.glob("*SPECTRAL_IMAGE.TIF"))
                      + list(input_dir.glob("*SPECTRAL_IMAGE.GEOTIFF"))
                      + list(input_dir.glob("*-SPECTRAL_IMAGE.TIF")))
    if not img_candidates:
        raise FileNotFoundError(f"No SPECTRAL_IMAGE.TIF found in {input_dir}")
    return img_candidates[0], xml_path


# ---------------------------------------------------------------------------
# ENVI writers  (unchanged from previous version)
# ---------------------------------------------------------------------------

def _write_envi_header(hdr_path: Path, meta: dict, ncols: int, nrows: int,
                       nbands: int, interleave: str, band_names=None,
                       wavelength=None, fwhm=None, bbl=None,
                       map_info=None, coord_sys=None, description: str = "") -> None:
    lines = ["ENVI"]
    if description:
        lines.append(f"description = {{{description}}}")
    lines += [
        f"samples = {ncols}",
        f"lines   = {nrows}",
        f"bands   = {nbands}",
        "header offset = 0",
        "file type = ENVI Standard",
        "data type = 4",
        f"interleave = {interleave}",
        "byte order = 0",
        "sensor type = EnMAP",
    ]
    if map_info:
        lines.append(f"map info = {map_info}")
    if coord_sys:
        lines.append(f"coordinate system string = {{{coord_sys}}}")
    if wavelength is not None:
        lines.append("wavelength units = Nanometers")
        lines.append("wavelength = {" + ", ".join(f"{w:.4f}" for w in wavelength) + "}")
    if fwhm is not None:
        lines.append("fwhm = {" + ", ".join(f"{f:.4f}" for f in fwhm) + "}")
    if bbl is not None:
        lines.append("bbl = {" + ", ".join(str(b) for b in bbl) + "}")
    if band_names is not None:
        lines.append("band names = {" + ", ".join(band_names) + "}")
    hdr_path.write_text("\n".join(lines) + "\n")


def _map_info_from_transform(src: rasterio.DatasetReader) -> str | None:
    t = src.transform
    crs = src.crs
    if crs is None:
        return None
    epsg = crs.to_epsg()
    if epsg and 32601 <= epsg <= 32660:
        zone = epsg - 32600
        return f"{{UTM, 1, 1, {t.c}, {t.f}, {t.a}, {-t.e}, {zone}, North, WGS-84, units=Meters}}"
    if epsg and 32701 <= epsg <= 32760:
        zone = epsg - 32700
        return f"{{UTM, 1, 1, {t.c}, {t.f}, {t.a}, {-t.e}, {zone}, South, WGS-84, units=Meters}}"
    if epsg == 4326:
        return f"{{Geographic Lat/Lon, 1, 1, {t.c}, {t.f}, {t.a}, {-t.e}, WGS-84, units=Degrees}}"
    return f"{{Arbitrary, 1, 1, {t.c}, {t.f}, {t.a}, {-t.e}, units=Meters}}"


def write_radiance_cube(img_path: Path, meta: dict, out_path: Path) -> None:
    with rasterio.open(img_path) as src:
        rows, cols, nbands = src.height, src.width, src.count
        assert nbands == len(meta["bands"]), (
            f"Band count mismatch: TIFF={nbands}, XML={len(meta['bands'])}"
        )

        wl = [b["wl"] for b in meta["bands"]]
        fwhm = [b["fwhm"] for b in meta["bands"]]
        bbl = [0 if any(lo <= w <= hi for lo, hi in BAD_BAND_RANGES_NM) else 1
               for w in wl]

        map_info = _map_info_from_transform(src)
        coord_sys = src.crs.to_wkt() if src.crs else None

        _write_envi_header(
            out_path.with_suffix(".hdr"),
            meta, cols, rows, nbands, "bil",
            wavelength=wl, fwhm=fwhm, bbl=bbl,
            map_info=map_info, coord_sys=coord_sys,
            description=f"EnMAP L1C radiance, {meta['scene_id']}, uW/cm^2/sr/nm",
        )

        nodata_in = (src.nodatavals[0]
                     if src.nodatavals and src.nodatavals[0] is not None else 0)
        with open(out_path, "wb") as fout:
            for row in range(rows):
                line = np.zeros((nbands, cols), dtype=np.float32)
                for i, band in enumerate(meta["bands"]):
                    dn = src.read(i + 1, window=rasterio.windows.Window(0, row, cols, 1))[0]
                    dn = dn.astype(np.float32)
                    valid = dn != nodata_in
                    rad = dn * band["gain"] + band["offset"]
                    rad *= RAD_UNIT_SCALE
                    # Clip valid pixels to a small positive floor to avoid
                    # zero/negative radiance breaking the OE noise model.
                    rad = np.where(np.isfinite(rad), rad, -9999.0)
                    rad = np.where(valid, np.clip(rad, 1e-4, None), -9999.0)
                    line[i] = rad
                line.tofile(fout)
                if row % 200 == 0:
                    log.info("  radiance: line %d/%d", row, rows)

    log.info("Wrote radiance cube: %s (%d x %d x %d)", out_path, rows, cols, nbands)


def _pixel_center_grid(src: rasterio.DatasetReader) -> tuple[np.ndarray, np.ndarray]:
    t = src.transform
    cols = np.arange(src.width) + 0.5
    rows = np.arange(src.height) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    xs = t.a * cc + t.b * rr + t.c
    ys = t.d * cc + t.e * rr + t.f
    return xs, ys


def _sample_dem(dem_path: Path, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    shape = lons.shape
    with rasterio.open(dem_path) as dem:
        if dem.crs and dem.crs.to_epsg() != 4326:
            xs, ys = warp_transform("EPSG:4326", dem.crs,
                                    lons.ravel().tolist(), lats.ravel().tolist())
            coords = list(zip(xs, ys))
        else:
            coords = list(zip(lons.ravel(), lats.ravel()))
        elev = np.fromiter(
            (v[0] if v[0] is not None else -9999.0 for v in dem.sample(coords)),
            dtype=np.float32, count=len(coords),
        )
    return elev.reshape(shape)


def write_location_cube(img_path: Path, dem_path: Path, out_path: Path,
                        meta: dict) -> np.ndarray:
    with rasterio.open(img_path) as src:
        rows, cols = src.height, src.width
        xs, ys = _pixel_center_grid(src)

        if src.crs and src.crs.to_epsg() != 4326:
            lon_flat, lat_flat = warp_transform(
                src.crs, "EPSG:4326", xs.ravel().tolist(), ys.ravel().tolist()
            )
            lons = np.array(lon_flat, dtype=np.float32).reshape(rows, cols)
            lats = np.array(lat_flat, dtype=np.float32).reshape(rows, cols)
        else:
            lons = xs.astype(np.float32)
            lats = ys.astype(np.float32)

        map_info = _map_info_from_transform(src)
        coord_sys = src.crs.to_wkt() if src.crs else None

        log.info("Sampling DEM at %d pixels...", rows * cols)
        elev = _sample_dem(dem_path, lons, lats)

        _write_envi_header(
            out_path.with_suffix(".hdr"),
            meta, cols, rows, 3, "bip",
            band_names=["longitude", "latitude", "elevation"],
            map_info=map_info, coord_sys=coord_sys,
            description=f"EnMAP L1C location cube, {meta['scene_id']}",
        )
        stack = np.stack([lons, lats, elev], axis=-1).astype(np.float32)
        stack.tofile(out_path)

    log.info("Wrote location cube: %s", out_path)
    return elev


def _slope_aspect_cosi(elev, xres_m, yres_m, sun_zenith_deg, sun_azimuth_deg):
    e = np.where(elev > -1000, elev, 0).astype(np.float32)
    dz_dy, dz_dx = np.gradient(e, yres_m, xres_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect_rad = np.arctan2(dz_dx, -dz_dy)
    aspect_rad = np.where(aspect_rad < 0, aspect_rad + 2 * np.pi, aspect_rad)
    sza = math.radians(sun_zenith_deg)
    saa = math.radians(sun_azimuth_deg)
    cos_i = (math.cos(sza) * np.cos(slope_rad)
             + math.sin(sza) * np.sin(slope_rad) * np.cos(saa - aspect_rad))
    # Clip away from zero so downstream divides (L_dir_dir / cos_i etc)
    # stay finite even on steep slopes facing away from sun.
    cos_i = np.clip(cos_i, 0.05, 1.0)
    return (np.degrees(slope_rad).astype(np.float32),
            np.degrees(aspect_rad).astype(np.float32),
            cos_i.astype(np.float32))


def write_observation_cube(img_path: Path, meta: dict, elev: np.ndarray,
                           out_path: Path) -> None:
    with rasterio.open(img_path) as src:
        rows, cols = src.height, src.width
        t = src.transform
        if src.crs and src.crs.is_projected:
            xres_m = abs(t.a); yres_m = abs(t.e)
        else:
            mean_lat = 0.5 * (t.f + (t.f + t.e * rows))
            xres_m = abs(t.a) * 111320.0 * math.cos(math.radians(mean_lat))
            yres_m = abs(t.e) * 111320.0
        map_info = _map_info_from_transform(src)
        coord_sys = src.crs.to_wkt() if src.crs else None

    sza = meta["solar_zenith"]; saa = meta["solar_azimuth"]
    vza = meta["view_zenith"];  vaa = meta["view_azimuth"]
    orbit_km = meta.get("orbit_altitude_km", DEFAULT_ORBIT_ALT_KM)

    cos_phase = (math.cos(math.radians(sza)) * math.cos(math.radians(vza))
                 + math.sin(math.radians(sza)) * math.sin(math.radians(vza))
                 * math.cos(math.radians(saa - vaa)))
    phase_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_phase))))

    # Sanitize DEM: replace non-finite / very-negative pixels with 0 m
    elev_clean = np.where(np.isfinite(elev) & (elev > -1000), elev, 0.0)
    # ISOFIT reads obs band 0 in METRES (template_construction.py and
    # core/geometry.py both apply m_to_km). Writing km put the sensor at ~0.7 km,
    # so 6S returned NaN path radiance for every LUT point above that height.
    path_m = 1000.0 * (orbit_km - elev_clean / 1000.0) / max(math.cos(math.radians(vza)), 0.1)
    path_m = np.where(np.isfinite(path_m), path_m, orbit_km * 1000.0)

    slope, aspect, cos_i = _slope_aspect_cosi(elev_clean, xres_m, yres_m, sza, saa)

    utc_hour = (meta["acquisition_time"].hour
                + meta["acquisition_time"].minute / 60.0
                + meta["acquisition_time"].second / 3600.0)

    bands = [
        path_m.astype(np.float32),
        np.full((rows, cols), vaa, dtype=np.float32),
        np.full((rows, cols), vza, dtype=np.float32),
        np.full((rows, cols), saa, dtype=np.float32),
        np.full((rows, cols), sza, dtype=np.float32),
        np.full((rows, cols), phase_deg, dtype=np.float32),
        slope, aspect, cos_i,
        np.full((rows, cols), utc_hour, dtype=np.float32),
    ]
    band_names = ["path_length_m", "to_sensor_azimuth", "to_sensor_zenith",
                  "to_sun_azimuth", "to_sun_zenith", "phase",
                  "slope", "aspect", "cosine_i", "utc_hour"]

    _write_envi_header(
        out_path.with_suffix(".hdr"),
        meta, cols, rows, 10, "bip",
        band_names=band_names,
        map_info=map_info, coord_sys=coord_sys,
        description=f"EnMAP L1C observation cube, {meta['scene_id']}",
    )
    stack = np.stack(bands, axis=-1).astype(np.float32)
    stack.tofile(out_path)
    log.info("Wrote observation cube: %s", out_path)


def write_wavelengths_file(meta: dict, path: Path) -> None:
    with open(path, "w") as f:
        for i, b in enumerate(meta["bands"], start=1):
            f.write(f"{i}\t{b['wl']:.4f}\t{b['fwhm']:.4f}\n")
    log.info("Wrote wavelength file: %s", path)


def guess_season(acq_time: datetime, lat: float | None = None) -> str:
    month = acq_time.month
    northern = (lat is None) or (lat >= 0)
    if northern:
        return "summer" if 4 <= month <= 9 else "winter"
    return "winter" if 4 <= month <= 9 else "summer"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--dem", required=True)
    p.add_argument("--workdir", required=True)
    args = p.parse_args(argv)

    input_dir = Path(args.input)
    dem = Path(args.dem)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    img_path, xml_path = find_scene_files(input_dir)
    log.info("Scene image:    %s", img_path.name)
    log.info("Scene metadata: %s", xml_path.name)

    meta = parse_metadata(xml_path)

    write_radiance_cube(img_path, meta, workdir / "rdn")
    elev = write_location_cube(img_path, dem, workdir / "loc", meta)
    write_observation_cube(img_path, meta, elev, workdir / "obs")
    write_wavelengths_file(meta, workdir / "wavelengths.txt")

    # Season hint from XML center or fallback to image transform
    center_lat = meta.get("center_lat")
    if center_lat is None:
        with rasterio.open(img_path) as src:
            t = src.transform
            cy = t.f + t.e * src.height / 2
            if src.crs and src.crs.to_epsg() != 4326:
                _, lat = warp_transform(src.crs, "EPSG:4326", [t.c], [cy])
                center_lat = lat[0]
            else:
                center_lat = cy
    (workdir / "season.txt").write_text(guess_season(meta["acquisition_time"], center_lat))

    log.info("Preprocessing complete. Inputs staged under: %s", workdir)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
