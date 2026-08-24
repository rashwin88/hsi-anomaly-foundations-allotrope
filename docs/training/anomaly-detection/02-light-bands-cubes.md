# Part 2 - Light, bands and cubes

> **The one thing this part teaches:** a hyperspectral pixel is not a colour, it is a curve,
> and that curve identifies what the pixel is made of.

## Start with an ordinary photo

A normal digital photo stores three numbers per pixel: red, green, blue. Your eye works the
same way - three kinds of colour receptor - so three numbers are enough to fool it.

Three numbers is very little information. Fresh grass and green paint can produce the same
RGB triple. Once the three numbers match, nothing distinguishes them.

## What a band is

**A band is a measurement of how much light a pixel reflects in one narrow range of
wavelengths.** Red is a band. So is "light between 1,650 and 1,660 nanometres", which your
eye cannot see at all.

Wavelength is measured in nanometres (nm). Visible light runs roughly 400 to 700 nm. The
sensors here go from 460 nm to 2,450 nm, so most of what they measure is invisible to you.

A **hyperspectral** sensor records 200 or more narrow bands instead of three wide ones.
PRISMA records 239.

## What that buys you

Materials absorb and reflect light differently at different wavelengths, and the pattern is
close to a fingerprint. Water absorbs strongly in the shortwave infrared. Vegetation
reflects sharply upward just past red - the "red edge" - because chlorophyll absorbs red and
leaf structure scatters near-infrared. Minerals have narrow absorption features at
wavelengths set by their chemistry.

With three bands, grass and green paint look identical. With 239, they are obviously
different, because paint has no red edge.

**One pixel's 239 numbers, plotted against wavelength, is called its spectrum.** The whole
system is built on comparing spectra.

## The cube

Stack a band for every wavelength and you get a three-dimensional array:

```
        width (W)
      +-----------+
     /           /|
    /           / |     each horizontal slice = one band
   +-----------+  |     (the whole scene at one wavelength)
   |           |  +
   |           | /      each vertical needle = one pixel's spectrum
   |           |/       (all wavelengths at one place)
   +-----------+
        height (H)   x   bands (C)
```

Two ways to read it, both correct and both used:

- **A stack of images.** 239 greyscale pictures of the same place, each at a different
  wavelength.
- **A grid of spectra.** One curve per ground location, 239 points long.

Detectors use the second reading. Visualisation uses the first.

## Thermal is the simple cousin

Landsat 9 and HotSat-1 measure **one** band - thermal infrared, which tells you how hot the
surface is. Same cube structure, `C = 1`.

You still need anomaly detection, because "hotter than its surroundings" is exactly what a
fire, a flare, or a piece of running machinery looks like. But there is no spectrum to
compare, so the techniques differ. Watch for that split: several things in this codebase
exist in a hyperspectral version and a thermal version.

## Reflectance, and why raw numbers are useless

A sensor does not record physics. It records **digital numbers** (DN) - integers proportional
to how much energy hit the detector, scaled by whatever the electronics do.

DN is not comparable across anything. Same ground, different sun angle, different DN. Two
sensors over the same field give different DN.

So the first processing step converts DN to a physical quantity:

- Hyperspectral: **surface reflectance**, the fraction of light reflected. A number between
  0 and 1, comparable across sensors and dates.
- Thermal: **surface temperature**, in Celsius.

For Landsat that conversion is two published constants, in
`app/utils/data_transformations/l2sp_dn_to_temperature_transformer.py`:

```python
SCALING_FACTOR: float = 0.00341802
ADDITIVE_FACTOR: float = 149.0
```

Work one through. A detector reports DN = 30,000:

```
kelvin  = 0.00341802 * 30000 + 149.0
        = 102.5406 + 149.0
        = 251.5406 K

celsius = 251.5406 - 273.15
        = -21.61 C
```

Minus 21.6 degrees - plausibly a cold cloud top. **Do that arithmetic yourself before
moving on.** The rest of the course assumes you are comfortable turning stored integers into
physical units.

## Common confusions

**"Is a band the same as a channel?"**
In practice yes. Remote sensing says band, deep learning says channel, and this codebase
uses both - often in the same file. A `(C, H, W)` tensor's `C` is the band count.

**"Hyperspectral means high resolution?"**
No, and this trips up nearly everyone. It means high *spectral* resolution - many narrow
wavelength ranges. Spatially, these sensors are coarse: a PRISMA pixel covers about 30
metres on the ground. You get a rich spectrum for a large patch.

**"So more bands is strictly better?"**
No. More bands means more noise per band, far more data, and a statistical problem you will
meet in part 8 where having many bands relative to usable pixels makes a key calculation
numerically unstable. There is a real reason one classical technique was banned outright
for hyperspectral data.

**"Reflectance between 0 and 1 - so values above 1 are bugs?"**
Usually, but not always. Specular glints off water or metal can exceed 1 after atmospheric
correction. Do not add a hard clamp assuming otherwise.

## Check yourself

<details>
<summary>1. What is a band, in one sentence, and how many does PRISMA have?</summary>

A measurement of reflected light in one narrow wavelength range. PRISMA has 239 - 66 in the
visible/near-infrared and 173 in the shortwave infrared.
</details>

<details>
<summary>2. A Landsat detector reports DN = 41,000. What surface temperature in Celsius? Show every step.</summary>

```
kelvin  = 0.00341802 * 41000 + 149.0
        = 140.13882 + 149.0
        = 289.13882 K

celsius = 289.13882 - 273.15
        = 15.99 C
```

About 16 degrees - ordinary ground on a mild day.
</details>

<details>
<summary>3. Why can three bands not separate grass from green paint, when 239 can?</summary>

Both reflect similarly in red, green and blue, so their RGB triples coincide. Across 239
bands, grass shows a sharp rise just past red - the chlorophyll red edge - which paint does
not have. The extra bands expose a feature the three wide ones average away.
</details>

<details>
<summary>4. A cube is 1,210 x 1,219 x 239 float32 values. How many bytes? Work it through.</summary>

```
values = 1210 * 1219 = 1,474,990 pixels
       * 239 bands   = 352,522,610 values
       * 4 bytes     = 1,410,090,440 bytes
                     = about 1.41 GB
```

One scene, one copy, in memory. This is why the codebase worries about how many copies of a
cube exist at once.
</details>

<details>
<summary>5. Why convert DN to reflectance at all, rather than detecting anomalies in DN?</summary>

DN depends on illumination, sensor and acquisition conditions, so it is not comparable
across scenes, dates or sensors. Reflectance is a physical property of the surface, which
lets one model work across sensors and lets a spectrum be compared against a laboratory
library. Part 14 depends on this entirely.
</details>

---

Next: [part 3](03-scene-project-action.md) - where you are in the product.
