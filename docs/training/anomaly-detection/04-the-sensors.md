# Part 4 - The five sensors

> **The one thing this part teaches:** five sensors hand you five incompatible things, and
> almost every awkward corner of the ingestion code exists to hide that.

## The roster

| sensor | kind | bands | you upload |
|---|---|---|---|
| PRISMA | hyperspectral | 239 | one `.he5` file |
| EnMAP | hyperspectral | 224 | a folder |
| AVIRIS-NG | hyperspectral | ~425 | a folder |
| Landsat 9 | thermal | 1 (band B10) | one `.tif` file |
| HotSat-1 | thermal | 1 | a folder |

Band counts drift as products are reprocessed. The structural differences below will not.

## They disagree about everything

**File shape.** PRISMA is a single HDF5 file with an internal directory tree. Landsat is a
single GeoTIFF. The other three are folders where the cube, the masks and the metadata are
separate files. So "the path to a scene" sometimes means a file and sometimes a directory -
resolved per sensor in `backend/allotrope/sensors/source_path.py`.

**Array layout.** The same cube can be stored three ways, and sensors disagree:

| layout | axes | who uses it |
|---|---|---|
| BIL, band-interleaved-by-line | `(H, C, W)` | PRISMA |
| BSQ, band-sequential | `(C, H, W)` | EnMAP, Landsat - and every model |
| BIP, band-interleaved-by-pixel | `(H, W, C)` | visualisation, spectral matching |

This is not pedantry. Read a BIL cube as BSQ and you get structured noise that looks like a
corrupted image. **If a cube ever looks like plausible-but-wrong static, check the axis order
before suspecting the data.** Conversions live in
`app/utils/image_transformation/image_cube_operations.py`.

The layouts are not arbitrary either. BIP puts one pixel's whole spectrum in contiguous
memory, so a per-pixel spectral operation is a fast linear walk. BSQ puts a whole band
contiguous, so a spatial convolution is. Choosing wrongly costs nothing in correctness and a
great deal in speed.

**Calibration.** PRISMA needs per-band scale factors read from file metadata. EnMAP uses one
uniform gain, `SR = DN * 0.0001`. Landsat uses two published constants. There is no shared
formula.

**Quality information.** EnMAP ships ready-made cloud, cirrus, haze, shadow and snow masks.
PRISMA ships none. Landsat ships a bit-packed `QA_PIXEL` band you must unpack yourself. That
asymmetry is why `BandFilterConfig.quality_masks_to_apply` exists and applies to EnMAP alone.

## How the code hides this

Two layers.

**`FileHelper`** wraps the physical file - `HE5Helper`, `TIFHelper`, `EnmapHelper`,
`ENVIHelper`, `HotSatHelper`. Its job is to hand back raw arrays and metadata.

**Templates** tell a helper where things live. A template is a dictionary mapping a logical
component to a physical location, using one of three retrieval strategies
(`app/models/hyperspectral_concepts/references.py`):

| strategy | means | example |
|---|---|---|
| `FILE_REFERENCE` | an internal path inside the container | PRISMA's `HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube` |
| `ROOT_METADATA_FIELD` | a root-level attribute | PRISMA's `List_Cw_Swir` wavelength list |
| `DIRECT_PROPERTY_DEFINITION` | a property on the opened dataset | rasterio's `crs` or `bounds` |

Templates are **injected, not looked up**. The helper never contains a sensor-specific path,
which is what lets one contract serve HDF5, GeoTIFF and ENVI.

Only PRISMA, Landsat and EnMAP have templates. AVIRIS-NG and HotSat bypass the system and
parse their own sidecars - an inconsistency, honestly noted in the source.

## Traps that will cost you an afternoon

**rasterio band indices are 1-based.** Every other array in this codebase is 0-based. Asking
for band 0 is an error; band 1 is the first one.

**PRISMA keeps its bad bands.** A band the sensor flagged as unusable still occupies a slice
in the cube. Those slices carry `FWHM = 0.0`. Filter on the validity flag before trusting
anything about a band.

**AVIRIS-NG cubes are not in this repository.** `samples/aviris_samples/` holds only the
ENVI `.hdr` headers and `.sha256` checksums - the actual `.bin` files are absent. You cannot
run the AVIRIS path locally without supplying them.

## FWHM, and why it matters later

Alongside each band's centre wavelength sits its **full width at half maximum** - how wide a
slice of the spectrum that band actually samples.

A band is not a knife-edge. A band centred at 1,650 nm with FWHM 10 nm responds to a range
around 1,650, most strongly at the centre and tapering off.

Part 14 needs this. Comparing a satellite spectrum against a laboratory one means simulating
how the lab spectrum would look through this sensor's bands, which requires their widths.
**A scene without FWHM values cannot be material-matched at all.**

## Common confusions

**"Why not convert everything to one format on ingest and forget sensors?"**
That is exactly what happens - the result is the vendable, part 5. This part is the mess
that conversion hides.

**"BIL, BSQ, BIP - is one correct?"**
No. Each is optimal for a different access pattern, and sensor vendors chose based on how
their instrument writes data.

**"If AVIRIS-NG has ~425 bands and PRISMA 239, how does one model handle both?"**
It does not, directly. Both are resampled onto a shared 165-band grid first - part 7.

**"Two sensors are thermal - are they interchangeable?"**
No, and this is a live trap. Landsat 9 delivers calibrated Celsius. HotSat-1 delivers raw
DN. A model whose normalisation was fitted on Celsius sees HotSat DN as wildly
out-of-distribution. Part 10 works the arithmetic.

## Check yourself

<details>
<summary>1. Why does "the path to a scene" mean different things for different sensors?</summary>

PRISMA and Landsat are single files; EnMAP, AVIRIS-NG and HotSat are folders holding the
cube, masks and metadata separately. `sensors/source_path.py` resolves the right thing per
sensor.
</details>

<details>
<summary>2. A cube arrives as (1210, 66, 1219). Which layout, and what are H, C and W?</summary>

BIL - `(H, C, W)`. Height 1210, 66 bands, width 1219. The band count sitting in the middle
is the giveaway; a BSQ cube would be `(66, 1210, 1219)`.
</details>

<details>
<summary>3. EnMAP DN is 4,250. What reflectance? And is that physically plausible?</summary>

```
SR = 4250 * 0.0001
   = 0.425
```

42.5% reflectance - plausible for bright bare soil or a light roof. Vegetation in the
near-infrared can reach this; vegetation in red would be far lower, around 0.05.
</details>

<details>
<summary>4. Why can a scene without FWHM values not be material-matched?</summary>

Matching compares a satellite spectrum against a lab spectrum, which requires simulating how
the lab spectrum would appear through this sensor's bands. That simulation convolves the lab
curve with each band's response function, and FWHM is that function's width. Without it
there is nothing to convolve with.
</details>

<details>
<summary>5. You ask for band 0 from a GeoTIFF via rasterio and get an error. Why, and what is the wider lesson?</summary>

rasterio band indices are 1-based; band 1 is the first. The wider lesson is that this
codebase mixes conventions at the boundary between remote-sensing libraries and array code,
so check the convention before assuming an off-by-one is a bug elsewhere.
</details>

---

Next: [part 5](05-the-vendable.md) - the structure that makes all five look alike.
