# Input formats and ingestion: agent brief

Context for an agent working on how `isofit-atm-docker` reads its inputs. It covers which sensors and file types
arrive, what is inside each, how ENVI and GeoTIFF cubes are read and checked, and what the reader layer hands
ISOFIT. Code: `src/readers/` (generic path), `src/enmap_l1c_preprocess.py`, `src/prisma_l1_preprocess.py` (native
paths). Design: `docs/lld/isofit-atm-docker.md`. Tests: `tests/test_cube_geometry.py`, `tests/test_prep.py`,
`tests/test_level.py`.

## 1. The contract: what every input becomes

Whatever arrives, `python -m readers.prep --input <file|folder> --workdir <dir>` writes the same set of files:

| File | Content |
|---|---|
| `<fid>_rdn` + `.hdr` | TOA radiance, **µW/cm²/sr/nm**, float32, **BIL**, `-9999` = invalid. Bands dead in > 5% of sampled pixels are dropped and listed |
| `<fid>_loc` + `.hdr` | lon, lat, elevation (m), float32, BIP |
| `<fid>_obs` + `.hdr` | ISOFIT's 10 observation bands, float32, BIP: path length, to-sensor az/zen, to-sun az/zen, phase, slope, aspect, cos i, UTC hour |
| `<fid>_rfl` | only when the input is already surface reflectance and passes the SR test: the cube passed through |
| `wavelengths.txt` | index, centre nm, FWHM nm |
| `scene.json` | format, sensor rule, level verdict and tests, centre lat/lon, date, pixel size, **grade A–D**, checklist (every fact with its source: metadata / derived / assumed / missing), notes |

ISOFIT's unit is µW/cm²/sr/nm. Every radiance conversion below ends there.

## 2. Routing: `readers.prep.detect()`

| What is passed | Detected as | Path |
|---|---|---|
| Folder with `*-METADATA.XML` + `*SPECTRAL_IMAGE*` | `enmap_l1c` | native preprocessor |
| A `*SPECTRAL_IMAGE*` file inside such a folder | `enmap_l1c` (its folder) | native. The image alone is raw counts; the gains are in the XML |
| Folder with `ang*_rdn_*_img.hdr` + `*_obs_ort.hdr` + `*_loc_ort.hdr` | `aviris_ng_l1` | native: symlinked as delivered |
| `PRS_L1*.he5` | `prisma_l1` | native preprocessor |
| Any other `.h5` / `.he5` | `tanager_h5` | generic, via `readers/tanager.py` |
| `.tif` / `.tiff` | `geotiff` | generic, via `cube.open_geotiff` |
| Anything with an ENVI header next to it | `envi` | generic, via `cube.open_envi` |
| Folder with exactly one cube | that cube | if several: stop, "pass it directly" |

The sensor is identified separately (`readers/sensors.py`), strongest evidence first:
1. **The file's own name.** Rules: `ENMAP01-`, `PRS_L1|2`, `angYYYYMMDDtHHMMSS`, `YYYYMMDD_HHMMSS_NN_4001` (Tanager),
   `ghost` / `orbital sidekick`. The folder name never counts.
2. **Header or tag text.**
3. **Spectral signature:**
   - AVIRIS-NG: 425 bands from < 385 nm.
   - Tanager: 426 bands from < 385 nm.
   - GHOSt: grid 396.8–2507.2 nm.
   - EnMAP: 210–230 bands from 412–426 nm.
   - PRISMA: 220–240 bands, FWHM > 8 nm.
4. **Otherwise `unknown`**, which gets the generic rule.

`--sensor` overrides the identification.

## 3. The sensors and their native products

### EnMAP L1C (DLR), a folder
- `*-METADATA.XML` and `*SPECTRAL_IMAGE.TIF`. Newer deliveries name it `*SPECTRAL_IMAGE_COG.tiff`, and both are
  accepted.
- **The image holds raw counts.** Radiance = DN × `GainOfBand` + `OffsetOfBand`, per band from the XML, in
  W/m²/sr/nm. ×100 gives µW/cm²/sr/nm.
- **The XML schema varies by processor version** (v00.05–v00.07, including a `level_X` root from v01.05.06). The
  parser tolerates all of them. Geometry comes from the XML, with fallbacks:
  1. alternate field names;
  2. the sun computed from time and location;
  3. a documented near-nadir view default.
- 224 bands, about 418–2445 nm. VNIR and SWIR overlap around 900–1000 nm, so wavelengths are not strictly
  monotonic there.
- A COG's per-band wavelength is **not in the TIFF**. It is in the sibling `METADATA.XML`, which
  `cube._enmap_xml_axis` reads when a GeoTIFF lacks wavelength tags. This also covers DLR L2A, used as a
  reference.

### PRISMA L1 (ASI), `PRS_L1_STD_OFFL_*.he5` (HDF-EOS5)
- Group `HDFEOS/SWATHS/PRS_L1_HCO`: `Data Fields/VNIR_Cube` and `SWIR_Cube`, each **(lines, bands, samples)**.
  Geolocation is `Geolocation Fields/Latitude_SWIR` / `Longitude_SWIR`.
- **File attributes:**
  - `List_Cw_Vnir|Swir` (centres), `List_Fwhm_*`, `List_Cw_*_Flags` (valid bands);
  - `ScaleFactor_*`, `Offset_*`;
  - `Product_StartTime`, `Sun_zenith_angle`, `Sun_azimuth_angle`.
- **Radiance = (DN / ScaleFactor − Offset) × 0.1**: W/m²/sr/µm → µW/cm²/sr/nm. A DN of 0 is fill.
- **VNIR and SWIR are merged in ascending wavelength.** VNIR is kept below 950 nm and SWIR from 950 nm, so the
  920–980 nm overlap is not doubled.
- **No per-pixel view angles in L1.** `prisma_l1_geometry.py` derives them from the satellite track.
- **The product stays on the swath grid** (no map). The fid is `PRS_<start yyyymmddHHMMSS>`, which apply_oe's
  `prisma` mode parses.
- **L2C / L2D, used only as references:** cubes are uint16, reflectance = `L2ScaleVnirMin + DN × (Max − Min) /
  65535`, 0 = fill. **L2D is orthorectified (map grid); L2C stays on the L1 swath.** Compare against L2C.

### AVIRIS-NG L1 (JPL), a folder of ENVI files
- **Files:** `ang<YYYYMMDD>t<HHMMSS>_rdn_<ver>_img` (+ `.hdr`; data `.bin` or no extension), `_loc_ort`,
  `_obs_ort`, `_glt`.
  - `ver` is v2m2 / v2m3 / v2s1 / v2s2.
  - `loc_ort` / `obs_ort` are ISOFIT's own loc / obs layout, so they are used as delivered.
- **Radiance** is already µW/cm²/sr/nm (header: "Measured Radiances in uW nm-1 cm-2 sr-1"). Float32, BIL, 425
  bands from 376 nm.
- **The instrument grid is rotated:** `map info` carries `rotation=` (the trial line is 90°), and GDAL's
  geotransform holds it (non-zero b/d terms). Keep it end to end; never resample.
- **v2 headers carry 480-entry `fwhm` lists for 425 bands.** Trim every per-band list (`fwhm`, `wavelength`, `bbl`,
  `smoothing factors`, `correction factors`) to `bands`.
- **Flight lines are long.** The trial line is 5,592 × 722 (4 Mpix, 6.9 GB), and deliveries run up to 14–16 GB.
  Never load a whole cube: read row blocks or memmaps.
- **JPL L2 `corr` (reference only):** 0–1 float, but the water bands hold junk (up to 5.1e4 at 1358 nm, about 190
  at 1854–1909 nm). Never judge its scale from all bands (`level.quiet_bands`).

### Tanager (Planet), `<YYYYMMDD_HHMMSS_NN_4001>_{basic|ortho}_{radiance|sr}_hdf5.h5`
- **Two kinds:**
  - **basic** = swath: group `HDFEOS/SWATHS/HYP`, per-pixel `Geolocation Fields/Latitude`, `Longitude`, `Time`.
  - **ortho** = UTM grid: group `HDFEOS/GRIDS/HYP`. It has no lat/lon datasets; the georeference comes from
    `HDFEOS INFORMATION/StructMetadata.0` (`UpperLeftPointMtrs`, `LowerRightMtrs`) and the `epsg_code` attribute,
    corner-registered.
- **`Data Fields`:**
  - `toa_radiance` or `surface_reflectance`, shaped **(bands, lines, samples)**;
  - attributes `wavelengths`, `fwhm` (sometimes stored as a string list), `_FillValue` (−9999), `Unit`, and
    optionally `good_wavelengths`;
  - `nodata_pixels`, a mask: 1 = nodata;
  - per-pixel `sun_zenith`, `sun_azimuth`, `sensor_zenith`, `sensor_azimuth`, `sensor_to_ground_path_length`.
- **Radiance is W/(m² sr µm) → ×0.1.** SR is 0–1, and its variable name alone makes it surface reflectance.
- **Every geometry fact is metadata**, so a Tanager radiance file grades A. 426 bands from about 380 nm.
- **Planet's SR file also carries `aerosol_optical_depth` and `column_water_vapour`.** These are Planet's
  retrievals: useful for comparison, not truth.
- **Tanager ENVI exports carry isolated wavelength typos** (e.g. 376.44 between 381 and 386 nm).
  `cube.fix_order` replaces an out-of-order band with its neighbours' midpoint and notes it.

### GHOSt (Orbital Sidekick): GeoTIFF, delivered inconsistently
- **Deliveries seen so far:**
  - SR (OSK's own QUAC);
  - TOA-like with **no CRS**;
  - **raw DN with no scale**.
  - The grid is 396.8–2507.2 nm. Dataset 3 drops 25 bands.
- **Wavelength units may be tagged `wavelenght_units`** (sic). The reader accepts both spellings.
- **No published noise model and no sensor altitude per delivery.** The reader assumes 500 km, nadir view, and
  ISOFIT's generic noise, which typically gives grade C or D.
- **On the air-gapped VM, GHOSt scenes have also been dressed as AVIRIS-NG folders** (memory: GHOSt air-gapped
  handover).

## 4. Reading ENVI (`cube.open_envi`)

1. **Find the header:** `<file>.hdr` or `<stem>.hdr`. The data file is the stem with no extension, or `.dat`,
   `.img`, `.bin`, `.raw`, `.bsq`, `.bil` or `.bip`.
2. **Dtype from `data type`:** 1 u1, 2 i2, 3 i4, 4 f4, 5 f8, 12 u2, 13 u4, 14 i8, 15 u8. Byte order comes from
   `byte order` (1 = big-endian), and the `header offset` is honoured. **File size is checked** against
   lines × bands × samples × itemsize; a short file fails loudly.
3. **Interleave is checked against the data, not trusted.** A BIL/BSQ/BIP mix-up reads back as plausible garbage
   rather than an error. `check_interleave` reads 8 mid-scene lines under each layout and scores each with
   `layout_score`:
   - spectral **coherence**: each pixel's spectrum against the block mean;
   - × the **weakest smoothness** along bands, samples and lines.

   It keeps the declared layout unless another scores clearly better (> 0.05), and notes any switch. This was
   tested on AVIRIS-NG radiance and reflectance written in all three layouts under all three headers: 9/9 correct.
4. **Invalid values** become NaN: `data ignore value`, non-finite values, and any value ≤ −9990 in the level sample
   (an undeclared −9999 fill).
5. **Wavelengths** come from `wavelength` plus `wavelength units`. Without units, values < 100 are taken as µm →
   ×1000. `fwhm` is converted to nm the same way. **No wavelengths → stop** with "pass `--wavelength-file`" (1, 2
   or 3 columns: `wl` | `wl fwhm` | `idx wl fwhm`, nm or µm, one row per band).
6. **`reflectance scale factor`** is recorded and used by the level step for integer reflectance.
7. **Georeference via rasterio/GDAL** (`_georef`), so `map info` with rotation and `coordinate system string` are
   handled by the same code as GeoTIFF. No CRS, or an identity transform, means no location.

## 5. Reading GeoTIFF (`cube.open_geotiff`)

1. **Bands are read through rasterio** in row windows (`read_rows`), returned as **(rows, bands, samples)**
   float32 with NaN for invalid values. Invalid = `nodata` or non-finite. GDAL scale/offset is applied if present,
   and noted.
2. **Wavelength per band**, first match wins:
   1. band tags `wavelength` / `wavelength_nm` / `center_wavelength`;
   2. the **band description**, parsed by regex `([0-9.]+)\s*(nm|um|µm|micrometers?|nanometers?)` (e.g.
      `"Band 12 (0.523 um)"`);
   3. the `ENVI` metadata domain (`wavelength = {…}`);
   4. a sibling EnMAP `METADATA.XML`;
   5. `--wavelength-file`.
   - Units come from band or dataset tags `wavelength_units` (or GHOSt's `wavelenght_units`); unitless values
     < 100 are µm.
3. **FWHM:** band tags `fwhm`, else the XML, else derived (`derive_fwhm`, the smaller neighbour spacing, so a
   band-gap edge keeps its true sampling). A derived FWHM is recorded as `derived`.
4. **Interleave:** taken from `IMAGE_STRUCTURE:INTERLEAVE`. Only the delivered layout is recorded; reads go through
   GDAL, so there is no layout ambiguity.
5. **Georeference:** the transform and CRS, if present and not identity. **Many GHOSt and research GeoTIFFs have
   none.** The reader then has no location: a pseudo lat/lon grid, grade D (see §7).

## 6. Band order, duplicates, typos (`cube.sort_bands`, `cube.fix_order`)
- **Descending deliveries** are reordered ascending, and the reader is permuted to match.
- **An isolated out-of-order value** (neighbours consistent) is replaced by the neighbours' midpoint, and noted.
- **Duplicate centres** (overlapping detectors, e.g. EnMAP VNIR/SWIR) are kept, and noted.

## 7. After reading: level, units, metadata ladder (`readers/prep.py`, `level.py`, `sensors.py`)
- **Level** (`level.classify`, on a pixel sample from 12 row blocks):
  - **radiance**, from the shape test: p90 of L2200/L550 < 0.3;
  - **`dn`**: integer counts with the sun's shape and no gain → **stop**, "pass `--radiance-scale`";
  - **TOA reflectance**;
  - **surface reflectance**, only if ≥ 3 atmosphere tests ran and all vote SR (O2 760, H2O 940 / 1135, 1380) → pass
    through;
  - **unknown** → **stop**, "pass `--input-level`".
  - Reflectance scale: 0–1, or ×1e-4 for integers, judged on quiet bands only.
  - **The rule is asymmetric:** TOA must never pass as SR.
- **Radiance units:** read from header or tag text (`units`, `radiance units`, `description`) by regex. µW/cm²/sr/nm
  = 1, W/m²/sr/µm = 0.1, W/m²/sr/nm = 100, mW/cm²/sr/µm = 1, and so on. Otherwise inferred from magnitude: the
  candidate scale whose implied TOA reflectance at 860 nm is plausible.
- **TOA reflectance → radiance:** uses ISOFIT's own solar irradiance (`prism_optimized_irr.dat`), so it round-trips
  inside ISOFIT.
- **Fact ladder:**
  - **time:** header/tags → file name.
  - **location:** CRS/map info, or per-pixel lat/lon (HDF5).
  - **sun:** NOAA formula from time + location.
  - **view:** delivered, else nadir (assumed).
  - **altitude:** path length, else nominal orbit.
  - **DEM:** the mounted VRT, else flat 0 m (assumed). With no DEM, pixel (0, 0) gets 3 km so ISOFIT's elevation
    grid spans 0–5 km.
  - **No location:** a pseudo 30 m lat/lon grid near (0, 0), never a constant. With a constant, ISOFIT's spatial
    interpolation zeroes AOT and H2O.
  - Each fact records its source; the grade summarises them.
- **Overrides**, each recorded as "set by the operator":
  - `--sensor`;
  - `--input-level radiance|toa|sr`;
  - `--radiance-scale X`;
  - `--wavelength-file F`.

## 8. Rules for anyone changing ingestion
- **Never trust an ENVI interleave or a GeoTIFF's band order** without the data checks above.
- **Never read a whole cube into memory.** Use row blocks, memmaps and strided reads. AVIRIS-NG lines reach 16 GB.
- **Never judge reflectance scale on all bands.** Vendors put junk in water bands (`level.quiet_bands`).
- **Never resample on ingest.** Swath products stay on the swath grid, and rotated grids keep their rotation.
- **A new fact source:** add it to the ladder with its `Fact` source, so `scene.json` and the grade stay honest.
- **A new format:** route it in `detect()`, return a `CubeInfo` (or a `Scene` via `native_record`), and add a
  synthetic-data test (`tests/conftest.py` has the generators: `write_tif`, `write_envi`, `write_tanager`). The
  build gate is ≥ 30% coverage, currently 54%.
