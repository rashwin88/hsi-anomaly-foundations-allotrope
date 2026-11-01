# Robust hyperspectral ingestion - any file, no code change during evaluation

**Status:** built 2026-11-01 as `sensor_type="generic_hsi"` - [lld/generic-hsi.md](../lld/generic-hsi.md); not yet on the air-gapped VM. Agreed 2026-11-01 (user: "When eval begins we can't change code, so we will have to make it robust to any
last-minute surprise"; "Yes please"). To be built after the Tanager ENVI path lands (it shares the GDAL reader).

## Why

Once AIGC evaluation starts the code is frozen. The evaluation data may be GHOSt (the competition data was), arrive
as single GeoTIFFs (as the competition's three GHOSt files did), as OSK's native ENVI, or as something else. Today
only PRISMA (.he5), EnMAP (folder), AVIRIS-NG (ENVI folder) and Tanager (HDF5, ENVI in progress) onboard; a GeoTIFF
cube is accepted only for thermal sensors, and GHOSt goes through an offline AVIRIS-NG costume
(`OneDrive/GAIC/ghost_handover/ghost_to_envi.py`) that mislabels the sensor, depends on BIL layout and exports nothing
without a map in the header.

## What - a `generic_hsi` sensor (GHOSt is its first user; no separate `ghost` sensor)

Native readers stay for PRISMA / EnMAP / AVIRIS-NG / Tanager (they use provider quality masks). The generic path is
the safety net for everything else.

1. **Formats:** anything GDAL reads - GeoTIFF / COG (one file), ENVI (`.hdr` + data: `.hsi` (OSK), `.img`, `.bsq`,
   `.bil`, `.bip`, `.dat`, none), HDF5 / NetCDF subdatasets (choose the 3-D reflectance-like one). Any interleave, via
   GDAL - never hand-built memmaps (the AVIRIS-NG path misreads BSQ silently).
2. **Wavelengths**, first hit wins:
   1. per-band metadata `wavelength` (also the misspelt `wavelenght_units` GHOSt's GeoTIFFs carry);
   2. the ENVI header `wavelength` / `fwhm` / `wavelength units`;
   3. HDF5 / NetCDF attributes;
   4. band descriptions such as `Band 1 (396.8 nm)`;
   5. **a wavelength file uploaded with the scene** (CSV or JSON), the escape hatch so no cube is un-ingestable.
   Order by wavelength, never by band index (GHOSt dataset 3: position 218 is labelled band 224). Micrometres to nm.
3. **FWHM:** from metadata when present (OSK headers carry `fwhm`, e.g. 8.7 nm). When absent: a sensor hint if the
   metadata names one (GHOSt: ~9 nm, per the OSK data sheet's "~10 nm FWHM" and the 8.7 nm example), else 1.95 x band
   spacing; always recorded as derived. (The GHOSt costume used 1.0 x spacing = 4.48 nm - about half the true width -
   which makes library spectra too sharp for matching.)
4. **Scale - detected, reported, overridable**, never guessed silently:
   - max <= ~1.5 -> reflectance 0-1;
   - median in the hundreds to low thousands and max ~<= 12000 -> reflectance x 10000;
   - a header `reflectance scale factor` / `scale_factor` / `add_offset` wins over the heuristic;
   - radiance units in the metadata, or anything else -> **"not reflectance"**.
   A not-reflectance scene onboards, but V2 and the namer show a warning and scoring defaults to MNF-RX (unit-free).
   The operator can override the decision (x1, x0.0001, radiance) at upload.
5. **No-data:** header value, GDAL nodata, 0, -9999 or NaN; validity also requires finite values.
6. **Georeferencing:**
   - GeoTIFF / ENVI map info / CRS -> exact grid; a geographic CRS (EPSG:4326) is a valid grid;
   - **OSK unrectified L2A / L1B + `.igm`** -> per-pixel lat/lon from the IGM, handled like PRISMA's swath;
   - none -> onboard and score, but **export refuses** with "this file has no map coordinates" (an unplaceable map
     must never be submitted).
7. **Pre-flight report** at upload (and on the scene page): bands, wavelength range, units, FWHM (read or derived),
   scale decision, no-data, georef source, data type, and every assumption made. The operator sees a surprise before
   running anything.
8. **Defaults:** band filter native bands with the usual exclusion ranges; scoring Indradhanu V2 + MNF-RX
   (MNF-RX only when not reflectance); namer on (warned when not reflectance).

## Proof before the freeze

A robustness suite of synthetic surprise files: each format and interleave, um vs nm, shuffled and descending
wavelengths, misspelt keys, missing FWHM, the 0-1 / x10000 / DN / radiance scales, every no-data form, geographic CRS,
no georef, IGM swath, mislabelled band descriptions. Every case runs band filter -> scoring -> threshold -> export (or
refuses export where it must), in tests and end to end on the local stack. Plus the real files we have: Tanager (HDF5
and ENVI) and the GHOSt GeoTIFF layout (dataset 1-3 shapes from `ghost_handover/gdalinfo_notes_GHOSt.txt`).

## Dependency

It only protects the evaluation if the **air-gapped VM receives this code before evaluation starts** - a deploy that
needs the user's explicit go and a time.

## Sources

- OSK Hyperspectral Product Data Sheet (2026), NASA CSDA:
  https://science.nasa.gov/wp-content/uploads/2026/08/osk-hyperspectral-product-data-sheet-2026.pdf. It gives: ENVI
  `.hsi` / `.hdr` / `.igm`; L1B, L1BR, L2A, L2AR; BSQ; `fwhm` 8.7 nm; 472 bands, 396.8-2507.2 nm. No reflectance scale
  factor or data type is stated.
- Competition GHOSt GeoTIFFs: `OneDrive/GAIC/ghost_handover/README.md`, `gdalinfo_notes_GHOSt.txt`.
