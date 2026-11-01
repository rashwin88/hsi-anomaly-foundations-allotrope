# L2 ingestion for anomaly detection: agent brief

You are hardening the ingestion of **Level-2 (surface reflectance) hyperspectral products** for Allotrope's anomaly
detection. The organisers supply L2 data for the anomaly task and want the result back as a **boolean anomaly map in
.geotiff and .shp**, on the input's own grid and georeference. Up to 3 submissions are allowed.

Your job is to read every L2 delivery correctly and say what you assumed. That means:
- **Native vendor formats:** EnMAP L2A, PRISMA L2C/L2D, AVIRIS-NG `corr`, Tanager SR.
- **GeoTIFF:** GHOSt deliveries, the least predictable.
- **ENVI:** Tanager `.dat` + `.hdr` exports, AVIRIS-NG.

For each delivery, get the reflectance values, the scale, nodata, the wavelengths and FWHM, the bands to trust, and
the georeference to write the answer back with.

The organisers' brief, for reference: "You all will be provided with separate Level-2 hyperspectral data to perform
anomaly detection exercises … submit maximum 3 results … the detected anomalies map in boolean, .geotiff and .shp
formats."

## 1. What already exists (read before writing code)

| Where | What | Use it for |
|---|---|---|
| `app/utils/dataset_builder/*_dataset_builder.py` | per-sensor readers into the vendable (EnMAP, PRISMA, AVIRIS-NG, Landsat, HotSat) | the integration point. Note what they assume about scale and nodata |
| `app/georef/from_scene_file.py` | recovers (transform, CRS) at export time: vendables carry **no** spatial reference | PRISMA L2D (EPSG:4326 affine fitted to corner lat/lon), EnMAP L2A (UTM GeoTIFF), AVIRIS-NG (ENVI map info). **No Tanager, no GHOSt yet** |
| `band_filter_apply` action | native bands → the **165-band common grid** (10 nm, 460–2450 nm, absorption windows excluded); fills invalid pixels | everything downstream (models, MNF-RX) sees this grid, so wavelengths must be right before it |
| `isofit-atm-docker/src/readers/` (`cube.py`, `tanager.py`, `level.py`, `sensors.py`) | numpy/rasterio/h5py-only readers, unit-tested (`isofit-atm-docker/tests/`), built for the atmospheric-correction stack | ENVI / GeoTIFF / Tanager-HDF5 reading, the interleave check, wavelength discovery, band-order repair, level and scale tests. **Port the logic** rather than importing across stacks (layering: `app/` imports nothing from `backend/` or the atm stack) |
| `isofit-atm-docker/INPUT_FORMATS.md` | the L1 version of this brief (radiance, native preprocessors) | detail on the generic readers |

## 2. Native L2 products

### EnMAP L2A (DLR), a folder
- **The image** is `*-SPECTRAL_IMAGE.TIF`, or `*SPECTRAL_IMAGE_COG.tiff` in newer deliveries. Accept both, by
  pattern, never by exact name. 224 bands, **int16 reflectance ×10000, nodata −32768**, UTM grid.
- **The COG carries no per-band wavelengths.** The spectral axis (centres, FWHM) is in the sibling
  `*-METADATA.XML`, in its per-band list. `isofit-atm-docker/src/enmap_l1c_preprocess.parse_metadata` reads it for
  L1C; verify the same elements on L2A. The schema varies by processor version, so parse tolerantly.
- **VNIR and SWIR overlap around 900–1000 nm.** Centres are not monotonic there, and duplicates are real.
- **Quality layers:** DLR products carry separate `*QL_*` quality files (cloud, snow, haze, … on L1C; check what the
  L2A folder actually holds). Where present, they are the only vendor masks you get, so use them to exclude clouds
  and snow from anomaly scoring.
- **DLR leaves some pixels nodata** (−32768). In our trials that included dark water in a coastal scene. Treat
  −32768 per pixel, not per band.

### PRISMA L2C / L2D (ASI), `PRS_L2C_STD_*.he5` / `PRS_L2D_STD_*.he5`
- **Layout:** HDF-EOS5, group `HDFEOS/SWATHS/PRS_L2C_HCO` or `PRS_L2D_HCO` (pick the swath that holds
  `Data Fields/VNIR_Cube`; do not take the first key). Cubes `VNIR_Cube`, `SWIR_Cube`, each **(lines, bands,
  samples)**, uint16.
- **Reflectance** = `L2ScaleVnirMin + DN × (L2ScaleVnirMax − L2ScaleVnirMin) / 65535`, the same for SWIR with its
  own attributes. **DN 0 = fill.**
- **Spectral attributes:** `List_Cw_Vnir` / `List_Cw_Swir` (centres), `List_Fwhm_*`, `List_Cw_*_Flags` (valid bands:
  apply them). Do not assume the attribute order is ascending: sort, then merge both families. Resolve the VNIR/SWIR
  overlap (~920–980 nm) by keeping one family per wavelength; the L1 preprocessor takes VNIR below 950 nm and SWIR
  from 950 nm.
- **Geolocation:** `Geolocation Fields/Latitude` and `Longitude`, per pixel.
  - **L2C stays on the L1 swath** (not map-projected).
  - **L2D is orthorectified.** `app/georef` fits an EPSG:4326 affine to L2D's corner lat/lon. That's fine for an
    overlay, but check the residual against the full lat/lon arrays before trusting it for a .shp.
  - On relief, L2D and L2C differ by many pixels: never mix them in one comparison.

### AVIRIS-NG L2 `corr` (JPL), ENVI
- **Files:** `ang<YYYYMMDD>t<HHMMSS>_corr_<ver>_img` + `.hdr` (data `.bin` or no extension); `_h2o_<ver>_img` is the
  retrieved water vapour, not reflectance.
- **Data:** float32, **0–1**, BIL, 425 bands from 376 nm.
- **Grid:** the instrument grid is rotated, so `map info` carries `rotation=`. GDAL handles it as a geotransform
  with non-zero b/d terms. **Keep the rotation all the way to the output .geotiff and .shp**; do not drop it to a
  north-up box.
- **Junk water bands:** JPL fills the absorption bands with junk, up to **5.1e4 at 1358 nm and about 190 at
  1854–1909 nm**. Any "is this ×10000?" test on all bands gets it wrong. Mask values outside [−0.2, 2.5], and judge
  scale only on quiet bands (§5).
- **Headers:** v2 headers can carry per-band lists (`fwhm`, `bbl`, …) longer than `bands` (480 for 425). Trim them
  to `bands`.
- **Size:** flight lines are long, 4–9 Mpix and up to 14–16 GB. Read row blocks; never load the cube whole.

### Tanager (Planet) SR
- **HDF5:** `<YYYYMMDD_HHMMSS_NN_4001>_{basic|ortho}_sr_hdf5.h5`.
  - The reflectance is `HDFEOS/{SWATHS|GRIDS}/HYP/Data Fields/surface_reflectance`, **(bands, lines, samples)**,
    0–1, `_FillValue` −9999. `nodata_pixels` is a mask (1 = nodata).
  - Wavelengths and FWHM are dataset attributes, sometimes stored as a string; parse numbers out of them.
  - The SR file also carries Planet's own retrievals (`aerosol_optical_depth`, `column_water_vapour`).
- **basic = swath:** per-pixel `Geolocation Fields/Latitude`, `Longitude`.
- **ortho = UTM grid:** no lat/lon arrays. The georeference is in `HDFEOS INFORMATION/StructMetadata.0`
  (`UpperLeftPointMtrs`, `LowerRightMtrs`) plus the group's `epsg_code`, corner-registered.
- **ENVI exports** (`*_ortho_sr*.dat` + `.hdr`): read as ENVI (§4). Known catch: **isolated wavelength typos**,
  e.g. 376.44 nm sitting between 381 and 386 nm. Repair an out-of-order value whose neighbours are consistent
  with their midpoint, and log it.
- **Missing from `app/georef` today:** add basic (swath lat/lon) and ortho (StructMetadata + EPSG).

## 3. GHOSt (Orbital Sidekick): GeoTIFF, three different deliveries

The organisers' GHOSt stage-3 files, from `gdalinfo`. They show every trap a GeoTIFF reader must handle:

| | Dataset 1 | Dataset 2 | Dataset 3 |
|---|---|---|---|
| Size (samples × lines) | 557 × 1179 | 392 × 1153 | 449 × 606 |
| Type | Float32 | **Float64** | **UInt16** |
| Bands | 472, 396.8–2507.2 nm (~4.5 nm) | 472 | **447**: 25 bands removed |
| CRS | WGS 84 (EPSG:4326) | **none**: corners in pixel units, e.g. (0, 1153), (392, 1153) | WGS 84 |
| Values (band 1) | 0–0.119, mean 0.069 | 0.088–0.146, mean 0.103, sd 0.004 | integers |
| What it is | **surface reflectance 0–1** (OSK ships QUAC) | **TOA-like**: high, flat blue = path radiance. **Not L2** | **counts / ×10000 with no stated scale** |
| Layout | COG, PIXEL interleave, 512 blocks, overviews | COG, `.aux.xml` sidecar with stats | PIXEL interleave |

Rules these files force:
1. **Wavelength per band from the band's own metadata,** not from its index:
   - the band tag `wavelength=…`, else the description `Band N (396.8 nm)`.
   - In dataset 3 the **description number is the original band number**, not the TIFF band index: TIFF band 218
     says "Band 224 (1396.000000 nm)". Index → wavelength must come from the tag/description value. Never compute
     it from a start + step.
   - Values come as `396.8` or `396.800000`; parse as float.
2. **Units key typo:** the units are tagged **`wavelenght_units=nm`** (sic). Accept `wavelength_units`,
   `wavelenght_units` and the ENVI-domain key. Missing units with values < 100 means µm.
3. **Nodata = 0** on every band:
   - A real reflectance of exactly 0 is rare but possible, so mark a pixel nodata only when **all** bands are 0
     or invalid.
   - Otherwise treat isolated zeros as invalid values in that band only.
4. **No CRS** (dataset 2): ingestion must still work.
   - Record "no georeference".
   - The output .geotiff then carries the same pixel grid and no CRS, and the .shp is in pixel coordinates. **Flag
     this in the submission notes**: the organisers can only overlay it on their own file.
5. **Check the level before scoring anomalies.** Dataset 2 is labelled with L2-style metadata but reads as TOA.
   - Run the level test (§5). If it fails, stop or flag; do not silently score TOA as SR.
   - Never correct it inside ingestion: that is the atmospheric-correction stack's job.
6. **Integer reflectance needs a scale decision** (dataset 3), made from the data (§5) and recorded.
7. **Float64** (dataset 2) halves the throughput. Cast to float32 after reading.
8. **Read COGs by window or block** (512 × 512), never whole-file.
9. **Ignore `.aux.xml` statistics** for anything but display: they are approximate (`STATISTICS_APPROXIMATE=YES`).

## 4. ENVI reading (Tanager exports, AVIRIS-NG, anything `.dat` / `.img` / `.bin` + `.hdr`)
- **Pair header and data:** `<stem>.hdr` with the data at `<stem>`, `<stem>.dat`, `.img`, `.bin`, `.raw`, `.bsq`,
  `.bil` or `.bip`. Also try `<file>.hdr`.
- **Dtype** from `data type`: 1 u1, 2 i2, 3 i4, 4 f4, 5 f8, 12 u2, 13 u4, 14 i8, 15 u8. Honour `byte order`
  (1 = big-endian) and `header offset`. **Check the file size** against lines × bands × samples × itemsize, and
  fail loudly if it's short.
- **Interleave: never trust the header alone.** A BIL/BSQ/BIP mix-up reads as plausible garbage, not an error.
  Read a few mid-scene lines under each layout and score spectral coherence × smoothness along
  bands/samples/lines (`cube.check_interleave` / `layout_score`). Keep the header's layout unless another is clearly
  better; log a switch.
- **Header keys to read:**
  - `wavelength` + `wavelength units`; `fwhm`, converted to nm the same way as the wavelengths;
  - `data ignore value`;
  - `reflectance scale factor`, which often sets the ×10000 decision;
  - `bbl` (bad-band list: 0 = bad);
  - `band names`;
  - `map info` (including `rotation=`) + `coordinate system string`, both read through GDAL/rasterio so they
    behave like GeoTIFF.
- **List lengths:** per-band lists longer than `bands` are trimmed; shorter ones mean the axis is unusable, so stop
  and ask for a wavelength file.

## 5. Decisions every delivery needs (do them in this order, log each)
1. **Wavelength axis.** Sources in priority order:
   1. band tags;
   2. band descriptions (regex `([0-9.]+)\s*(nm|um|µm|micrometers?|nanometers?)`);
   3. the ENVI header or ENVI metadata domain;
   4. the vendor sidecar (EnMAP XML, PRISMA attributes, Tanager attributes);
   5. an operator wavelength file (1, 2 or 3 columns: `wl` | `wl fwhm` | `idx wl fwhm`, one row per band).

   Then:
   - convert units;
   - sort ascending, permuting the data to match;
   - repair isolated typos;
   - keep real duplicates (overlapping detectors), noted.
2. **FWHM:** delivered, else derived as the smaller neighbour spacing, recorded as derived. Don't average across a
   band gap.
3. **Bad bands:** the union of
   - vendor flags (`bbl`, PRISMA `List_Cw_*_Flags`, Tanager `good_wavelengths`);
   - all-invalid bands;
   - bands outside the water windows' quiet range when the vendor fills them with junk.

   The 165-band grid excludes the absorption windows anyway, but junk must not leak into gap filling.
4. **Nodata:**
   - the delivered value (−9999, −32768, 0, `data ignore value`, `_FillValue`), plus NaN/inf, plus vendor masks;
   - out-of-range reflectance (< −0.2 or > 2.5 after scaling) is invalid.
   - **A pixel is nodata when all its trusted bands are invalid.** Otherwise only the bad values are, so a band
     filter can fill them.
5. **Level and scale:**
   - **Is it really surface reflectance?** Radiance has the sun's shape (p90 of L2200/L550 < 0.3); reflectance does
     not.
   - **TOA vs SR** comes from absorption depths on the scene median: O2 760, H2O 940 / 1135, 1380 (`level.classify`).
   - **Scale:** 0–1 if the 99th percentile on *quiet bands* is < 5, ×1e-4 if < 2e4, else unknown → stop.
     **Quiet bands** exclude < 420, 1340–1460, 1780–1990 and > 2450 nm: vendors put junk there.
   - A delivery that fails the SR test is flagged, never scored silently.
6. **Georeference** (needed for the submission):
   - map grid: CRS + transform, keeping any rotation;
   - swath: per-pixel lat/lon (PRISMA L2C, Tanager basic); write the boolean .geotiff on the swath grid with GCPs
     and a lat/lon sidecar, and the .shp from the lat/lon of each anomaly polygon's pixels;
   - none (GHOSt dataset 2): pixel coordinates, flagged.
   - **Never resample on ingest.** The anomaly map must sit on the delivered grid, pixel for pixel.
7. **Record everything** in a per-scene provenance record. That means:
   - each fact and its source (metadata / derived / assumed);
   - the scale, nodata rule and dropped bands;
   - level-test votes;
   - georeference mode;
   - any repairs.

   This is what lets a wrong submission be explained.

## 6. Output side (what ingestion must preserve for the submission)
- **Boolean anomaly .geotiff:** uint8 0/1 (or a 1-bit-like uint8), same width, height, CRS and transform as the
  input (GCPs for swath), nodata where the input was nodata. Note the organisers asked for "boolean"; state the
  encoding in the submission.
- **.shp:** polygons of anomaly pixels in the input's CRS (lat/lon from the swath arrays for swath products). For
  rotated grids, polygon vertices come from the rotated transform, not a north-up box.
- Both depend on step 6 being right at ingestion.

## 7. Tests to add (synthetic data, no network; see `isofit-atm-docker/tests/conftest.py` for generators)
- GeoTIFF:
  - with `wavelenght_units`;
  - with descriptions whose number differs from the band index (dataset 3);
  - float64;
  - nodata 0 with a few real zeros;
  - no CRS.
- ENVI:
  - each interleave under a wrong header;
  - over-long `fwhm` list;
  - `bbl`;
  - `reflectance scale factor`;
  - a rotated `map info`.
- Junk water bands do not flip the scale; a TOA cube is not accepted as SR.
- PRISMA-like HDF5 merge (descending VNIR, overlap) and Tanager-like HDF5 (string attributes, `nodata_pixels`).
- Georeference round trip: input → boolean .geotiff/.shp → same grid and CRS (rotation kept).
