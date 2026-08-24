# 04 · The vendable — the object the model is handed

> **The one thing this part teaches:** raw satellite files never reach the model.
> Everything downstream consumes one standardised object called a *vendable*.

---

## Why an intermediate object exists at all

Raw satellite files are genuinely unpleasant:

- Every vendor invents its own file layout.
- Quality flags are packed into bits, and the bit meanings differ per satellite.
- Some ship reflectance, some ship raw counts, some ship radiance.
- Detector boundaries, dead columns and calibration quirks all differ.

If every piece of code downstream had to cope with all of that, every piece of
code would need updating whenever a satellite was added.

So the very first thing this system does with a new scene is convert it, once,
into a clean standard object. That object is called a **vendable**.

> **Why the odd name?** Think "the thing we can *vend*" — dispense, hand out —
> to anything that asks. It is the standardised product of the messy intake
> process.

The rule is absolute:

> **A vendable is the currency of the system. Every detector, every model, every
> analysis step consumes one. Nothing downstream ever reads a raw sensor file.**

The classes are in
[`app/models/dataset/vendables.py`](../app/models/dataset/vendables.py).

---

## What is inside a hyperspectral vendable

Class name: `VendableHyperspectralDataset`. (EnMAP has a subclass with five
extra quality masks; PRISMA and AVIRIS-NG use this one.)

| Field | Type / shape | Meaning |
|---|---|---|
| `normalized_hyperspectral_cube` | array `(C, H, W)` | the reflectance values — **this is what the model sees** |
| `validity_cube` | array `(C, H, W)` | 1 = real measurement, 0 = not |
| `band_cw_order` | list of C floats | the centre wavelength of each band, in order |
| `band_fwhm_order` | list of C floats | how wide each band is |
| `spectral_family_order` | list of C enums | which of the five clean stretches each band is in |
| `band_validity_by_position` | list of C ints | per-band validity flags |
| `band_level_validity_score` | list of C floats | percentage of valid pixels in each band, 0–100 |

The first two are the ones that matter for this course. The rest is bookkeeping
that later steps (export, spectral matching) rely on.

> **"cw" means centre wavelength.** "fwhm" means full width at half maximum — a
> standard way of saying how wide a band is. You will not need either for the
> model itself.

The thermal cousin, `VendableThermalDataset`, has
`normalized_thermal_cube` (in degrees Celsius) instead, plus a cloud mask. When
you read the scoring code you will see it branch on exactly this difference.

---

## Two facts that surprise everyone

### Surprise 1 — a vendable has no idea where on Earth it is

There is no coordinate system field. No map projection. No latitude or
longitude. Look for a `transform` field; there is not one.

Every image file that any analysis step writes has what is called an **identity
transform**, which is the geographic equivalent of "pixel (0,0) is at position
(0,0) and each pixel is 1 unit wide". Meaningless as geography.

The real geography is reattached **only at the very end**, at export time, by
going back and re-reading the original raw file (the code for that is in
`app/georef/`).

Why on earth would you do it that way? Because carrying geographic metadata
through fifteen processing steps means fifteen chances to corrupt it. Deriving
it once, at the end, from the original source, means it is always right.

> **Practical consequence.** Do not add a transform field to the vendable
> expecting it to be filled in. Do not trust the coordinates in an intermediate
> `.tif`. They are placeholders by design.

### Surprise 2 — vendables are pickled, so class names are a wire format

The `band_filter_apply` step writes the vendable to disk as
`filtered_vendable.pkl` using Python's `pickle` module. Later steps read it back.

Here is the trap: **pickle stores the full class path** — something like
`app.models.dataset.vendables.VendableHyperspectralDataset` — inside the file.
When you load it, Python imports that exact path.

So if somebody renames the class, or moves the module, every vendable already
sitting on disk becomes unloadable. Not corrupted. Just unreadable, because the
path recorded inside no longer exists.

> **Treat the class names and module path in `vendables.py` as a published
> format, like a database schema or an API contract.** Renaming for tidiness is
> a breaking change.

---

## How a vendable actually reaches Indradhanu

There are two vendables in a scene's life, and the model uses the second one.

```
raw satellite file
      |
      | scene onboarding
      v
 onboarding vendable          (all the sensor's native bands)
      |
      | band_filter_apply Action
      v
 filtered vendable            (exactly 165 bands, common grid)
      |
      v
   Indradhanu
```

Here is the actual code that loads it, from
[`_anomaly_scoring_run.py`](../backend/allotrope/action_types/_anomaly_scoring_run.py):

```python
bf_dir = ctx.resolve_action_output(bf_output_id)
pickle_path = bf_dir / "filtered_vendable.pkl"
...
cube_np     = vendable.normalized_hyperspectral_cube      # (165, H, W)
validity_np = vendable.validity_cube                      # (165, H, W)
wavelengths = np.asarray(vendable.band_cw_order)          # (165,)
```

And then immediately, one important line:

```python
spatial_valid = (validity_np.sum(axis=0) > 0).astype(np.uint8)   # (H, W)
```

Unpacking that slowly:

- `validity_np` is `(165, H, W)` — a 0/1 value for each band of each pixel.
- `.sum(axis=0)` adds up along the band axis, giving `(H, W)`. If a pixel is
  valid in all 165 bands, its total is 165. If valid in none, its total is 0.
- `> 0` turns that into "was this pixel valid in **at least one** band?"
- Result: a flat `(H, W)` map of which pixels are usable.

Since hyperspectral validity is all-or-nothing per pixel (part 02), "valid in at
least one band" and "valid in band 0" give the same answer. The training code
uses the band-0 shortcut for speed; the scoring code uses the sum. Same result,
different trade-off.

---

## What band_filter_apply does, in one paragraph

You do not need its internals to understand Indradhanu, but you should know what
it is for.

It runs an eight-stage pipeline driven by a `BandFilterConfig`:

1. drop bands the sensor itself flagged as bad,
2. drop the atmospheric exclusion ranges (part 03),
3. trim unreliable bands at detector edges,
4. drop bands where too few pixels are valid,
5. …and finally, resample whatever survived onto the 165-band common grid.

It also fills small holes using the nearest valid pixel. Output:
`filtered_vendable.pkl`.

**If it has not run, Indradhanu has nothing to eat.** The code says so
explicitly:

```python
if not bf_output_id:
    raise ValueError(
        "anomaly_scoring on a hyperspectral scene needs "
        "input_band_filter_output_id."
    )
```

Thermal scenes skip this entirely and read the onboarding vendable directly —
there are no bands to filter when you only have one.

---

## The contract, stated exactly

By the time Indradhanu is invoked, it has been handed precisely two things:

```
scene_tensor : float32, shape (165, H, W)    reflectance, on the common grid
mask_tensor  : float32, shape (1,   H, W)    1.0 = valid pixel, 0.0 = not
```

Every part from 09 to 22 is about what happens to those two arrays.

Notice what is **not** in the contract: no wavelengths, no geography, no sensor
name, no metadata. The model does not know or care which satellite took the
picture. That is the payoff of the common grid.

---

## Common confusions

**"Is the vendable the file on disk, or the object in memory?"**
Both, at different moments. It is a Python object; `filtered_vendable.pkl` is
that object pickled to disk.

**"`normalized_hyperspectral_cube` — is that the same normalisation as part 09?"**
No, and this is a genuine naming collision. The `normalized_` prefix here means
"cleaned up and standardised by the intake pipeline" (units, layout, grid). The
z-score normalisation in part 09 is a separate thing that happens **inside** the
model, at run time.

**"Why does the validity cube have 165 identical layers?"**
Because the type is shared with earlier pipeline stages where it genuinely
varied per band. By this point they agree.

---

## Check yourself

1. Why does the system convert raw files into vendables at all?
2. Where does the geographic information live, and when is it reattached?
3. What breaks if you rename `VendableHyperspectralDataset`?
4. Which Action must run before Indradhanu can score a hyperspectral scene, and
   what file does it write?
5. Exactly what two tensors does the model receive, with shapes?

<details>
<summary>Answers</summary>

1. So that no downstream code ever has to understand vendor-specific raw
   formats. One standard object, many consumers.
2. Nowhere in the vendable. It is recovered at export time by re-reading the
   original raw file (`app/georef/`). Intermediate GeoTIFFs carry identity
   transforms.
3. Every already-pickled vendable on disk becomes unloadable, because pickle
   stored the old class path inside the file.
4. `band_filter_apply`, which writes `filtered_vendable.pkl`.
5. `scene_tensor` of shape `(165, H, W)` and `mask_tensor` of shape `(1, H, W)`.

</details>

---

**Next:** where this all sits in the wider product, in
[05-where-it-lives.md](05-where-it-lives.md)
