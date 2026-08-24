# Part 5 - The vendable

> **The one thing this part teaches:** a vendable is a normalised cube plus a record of
> which pixels are real, and every component downstream consumes one.

## The idea

Part 4 was five incompatible formats. A **`DatasetBuilder`** reads one of them and produces a
**`VendableDataset`** - the single structure everything downstream understands.

The word is odd. Read it as "the thing this scene can vend" - what it hands out once it is
cleaned up. Defined in `app/models/dataset/vendables.py`.

```
raw file ──► FileHelper ──► DatasetBuilder ──► VendableDataset ──► everything else
             (reads bytes)  (converts units,
                             filters bands)
```

After this point, nothing needs to know what a `.he5` file is.

## Three variants

| class | sensors | cube holds |
|---|---|---|
| `VendableHyperspectralDataset` | PRISMA, AVIRIS-NG | reflectance |
| `VendableEnmapHyperspectralDataset` | EnMAP | reflectance + five quality masks |
| `VendableThermalDataset` | Landsat 9, HotSat-1 | temperature in Celsius |

A hyperspectral vendable carries the cube, the validity cube, the wavelength of each band,
each band's FWHM, and per-band validity flags. Bands travel with their own metadata, so a
detector never has to guess what band 47 means.

## Validity, which will follow you everywhere

**The second most important field is the one that says which pixels are real.**

Every cube ships with a `validity_cube` of the same shape. It marks dead detectors, cloud,
off-swath edges - anything not a genuine measurement.

Ignore it and you will not get subtly worse results, you will get confidently wrong ones.
Invalid pixels commonly hold zero or a fill sentinel. A detector looking for unusual values
finds a huge region of exactly-zero reflectance, decides it is the most anomalous thing in
the scene, and ranks it top. **The most common source of false detections in this codebase
is forgetting a mask.**

## Validity is binary, on purpose

You might expect a pixel valid in 140 of 165 bands to be partly usable. It is not. Stage 6
of the band pipeline (part 7) forces an all-or-nothing decision: if more than 40% of a
pixel's bands are invalid, the whole pixel is marked invalid.

That threshold is `max_invalid_voxel_fraction = 0.4`. Everything downstream can then assume a
pixel either has a complete spectrum or has nothing, which removes an entire category of
special-casing from every detector.

## The trap: no spatial reference

**A vendable does not know where on Earth it is.**

No latitude, no longitude, no map projection. Every GeoTIFF an Action writes carries an
identity transform - a placeholder saying "pixel (0,0) is at coordinate (0,0)".

That looks like an oversight. It is deliberate. Georeferencing is recovered at the very end,
at export time, by re-opening the original raw file (`app/georef/`). Analysis needs pixel
grids; only the final deliverable needs map coordinates.

The practical consequences:

- Do not add a transform field expecting it to be populated.
- Do not open an intermediate GeoTIFF in a GIS and conclude the georeferencing is broken.
- A scene whose georeferencing cannot be resolved fails at *export*, not at analysis, with
  `422 crs_missing`.

## The second trap: vendables are pickled

`band_filter_apply` writes `filtered_vendable.pkl` and the api reads it back.

Python's pickle stores the **class path** - `app.models.dataset.vendables.VendableHyperspectralDataset` -
not the class definition. Rename or move any class in that module and every stored vendable
on disk becomes unreadable.

**Treat the class names and the module path as a wire format.** The module's own docstring
says so. This is the single sharpest constraint on refactoring in `app/`.

## Two cloud masks, not one

A thermal vendable carries both a `cloud_mask` predicted from the temperature distribution
and the provider's `QA_PIXEL`-derived masks, kept separately rather than merged.

The reason is that they disagree usefully. The provider's mask is authoritative but
conservative, and is not always present. The modelled one is derived from this scene's own
temperatures and is always available. Keeping both lets a consumer choose; merging would
throw away the disagreement, which is often the interesting part.

## Common confusions

**"Is a vendable the same as the raw file?"**
No. Raw is DN in the sensor's own layout. A vendable is physical units, a standard layout,
and a validity record. The raw file stays on disk for georeferencing at export.

**"Why is the thermal variant separate rather than a 1-band hyperspectral?"**
Different units - Celsius, not reflectance - and different extra fields. Forcing them
together would give one class where half the fields are always `None`.

**"validity_cube is (C, H, W)? Isn't (H, W) enough if validity is binary per pixel?"**
It is per-band-per-pixel, because band filtering needs to know which *bands* failed where.
After stage 6 the spatial pattern is uniform across bands, but the shape is kept.

**"Can I just add a field to a vendable?"**
Adding is safe. Renaming or moving one breaks stored pickles. Note also that HotSat's builder
passes a `units` field the model does not define - Pydantic silently discards it, so the
intended guard against treating DN as Celsius is not actually in effect. Source over
comments, again.

## Check yourself

<details>
<summary>1. What two things does a vendable carry, and why is the second one not optional?</summary>

A normalised cube in physical units, and a validity record. Without validity, invalid pixels
holding zeros or fill values get scored as extreme outliers and dominate the results.
</details>

<details>
<summary>2. A pixel is invalid in 55 of 165 bands. Valid or invalid after the pipeline? Show the arithmetic.</summary>

```
fraction invalid = 55 / 165
                 = 0.3333

threshold        = 0.4

0.3333 < 0.4  ->  pixel is kept
```

The 55 bad bands get filled by interpolation. Had it been 70 bands, `70/165 = 0.424 > 0.4`,
and the whole pixel would be invalidated.
</details>

<details>
<summary>3. You open an intermediate anomaly_score.tif in QGIS and it lands at (0,0) off West Africa. Bug?</summary>

No. Vendables carry no spatial reference, so Action-written GeoTIFFs have an identity
transform. Georeferencing is recovered at export by re-reading the raw scene.
</details>

<details>
<summary>4. Why is renaming a class in vendables.py more dangerous than renaming one elsewhere?</summary>

Vendables are pickled to disk and pickle records the class path. A rename makes every stored
`filtered_vendable.pkl` unreadable, breaking Actions that reference earlier outputs.
</details>

<details>
<summary>5. Why keep two cloud masks instead of combining them?</summary>

They disagree usefully. The provider mask is authoritative but conservative and sometimes
absent; the modelled one is always available and scene-adaptive. Merging discards the
disagreement, which is often informative.
</details>

---

Next: [part 6](06-anomaly-as-surprise.md) - the core idea the whole system rests on.
