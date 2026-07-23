# 4. DN → Surface Reflectance: EnMAP L2A

The EnMAP transformer is the simplest calibration in the codebase. Where PRISMA needs per-family scales read from the file header, EnMAP needs a single hard-coded gain because the DLR ground segment harmonizes the dynamic range during processing.

The implementation lives in [`enmap_l2a_dn_to_surface_reflectance_transformer.py`](../../app/utils/data_transformations/enmap_l2a_dn_to_surface_reflectance_transformer.py).

---

## 4.1 What the code does

```mermaid
flowchart LR
    A[Input cube int16] --> B[Pre-allocate output float32]
    B --> C[numexpr.evaluate input * 0.0001 + 0.0]
    C --> D[Mask sentinel -32768 to 0]
    D --> E[Return reflectance cube]
```

### Step-by-step

1. **Class constants.** `GAIN = 0.0001`, `OFFSET = 0.0` ([enmap_l2a_…py:26](../../app/utils/data_transformations/enmap_l2a_dn_to_surface_reflectance_transformer.py)).
2. **In-place vectorized multiply-add.** `numexpr.evaluate("input_array * gain + offset", out=output)` ([enmap_l2a_…py:60](../../app/utils/data_transformations/enmap_l2a_dn_to_surface_reflectance_transformer.py)).
3. **Sentinel handling.** DLR's L2A nodata sentinel is `-32768` (the minimum int16 value). Voxels equal to that value are zeroed in the output at [enmap_l2a_…py:65](../../app/utils/data_transformations/enmap_l2a_dn_to_surface_reflectance_transformer.py).

### Call sites

Invoked once per scene inside `EnmapDatasetBuilder.vend_dataset`. Because EnMAP delivers a single integrated cube (no SWIR/VNIR file split), there is no per-family loop.

---

## 4.2 Theory in plain language

The mapping is a single uniform linear rescale:

$$\rho = \text{DN} \cdot 10^{-4}$$

The DLR processing chain that produces L2A:

1. Inverts the on-board ADC.
2. Applies absolute radiometric calibration to convert digital number to top-of-atmosphere radiance.
3. Runs an atmospheric correction (MODTRAN-based) to invert the atmospheric transmission and convert to surface reflectance.
4. Quantizes the result back to a 16-bit integer using a single gain factor of $10^{-4}$ across all 224 bands.

The reason it can use one gain (where PRISMA needs two) is that step 3 already harmonizes the dynamic range. By the time the data is quantized, the per-detector differences have been absorbed into the radiometric correction.

### Why signed int16 instead of uint16

Atmospheric correction can over-shoot. Deep water-vapor bands at 1400 nm, for example, have very little surface signal — most of what the sensor sees is atmospheric scattering plus stray light. When MODTRAN subtracts the modeled atmospheric contribution, the residual surface reflectance can be slightly negative.

A negative reflectance is physically meaningless (you cannot reflect *more* than zero light), but recording it as a small negative number rather than clipping to zero preserves information: a downstream model can see that "the correction overshot by 0.002" and decide that this band is unreliable for this pixel.

A signed integer encoding lets the file carry these negative residuals without losing them. The dynamic range is $[-3.2768, 3.2767]$ in reflectance space — far more than any physical surface needs, but the extra headroom is essentially free.

### Why values can exceed 1

Reflectance values greater than 1 are physically impossible for diffuse surfaces. But L2A occasionally shows values like 1.2 or 1.5 over:

- Thin clouds (where multiple scattering and forward scattering can briefly exceed the "diffuse" approximation).
- Snow (where bidirectional reflectance distribution function effects matter).
- Sun glint on water (specular reflection, not diffuse).

The transformer does not clip these. The downstream validity mask is responsible for refusing to train on unphysical pixels, not the calibrator. Calibration is a measurement-level operation; physics policing is a modeling-level operation.

---

## 4.3 Worked numerical example

A few representative values:

```text
DN:     [   12,   500,   7500,  10000,  -32768]
ρ:      [0.0012, 0.0500, 0.7500, 1.0000,  0.0  ]   # nodata replaced with 0
```

The reflectance 0.75 at DN=7500 is unremarkable — bright vegetation in summer. The reflectance 1.0 at DN=10000 is at the physical-plausibility boundary; this would be a snow or cloud pixel, and the validity mask should flag it for downstream attention.

### A second variation: deep water absorption

In a SWIR band at 1900 nm over a deep water-absorption window, atmospheric correction overshoots. The DN cube might look like:

```text
DN at 1900 nm pixel:  [-450, -200, -50, 0, 25, 100]
ρ after transform:    [-0.045, -0.020, -0.005, 0.000, 0.0025, 0.010]
```

The negative reflectances are absurd but not flagged as nodata (since they are not `-32768`). The wavelength-exclusion stage of the `SpectralBandFilter` (Section 9) drops the 1900 nm band entirely for being a water-vapor window, so these values never reach the model.

---

## 4.4 Edge cases

| Situation                                  | Behavior                                                |
|--------------------------------------------|---------------------------------------------------------|
| DN exactly at `-32768`                     | Output set to 0; validity mask should flag (caller's responsibility) |
| DN at the int16 maximum `32767`            | Output is $32767 \times 10^{-4} = 3.2767$, which is absurd; should be flagged by validity |
| Negative DN that is not `-32768`           | Negative reflectance is produced as-is; downstream is expected to handle |
| Input is float instead of int16            | Multiply still works; the sentinel check uses `!= -32768.0` which may miss float-rounded sentinels |

The third row is the most likely place to be surprised: a slight negative reflectance is not flagged by the transformer. The validity signal must come from another source (the EnMAP XML's QA bands, or a downstream coverage check).

---

## 4.5 Comparison with PRISMA

| Aspect                          | PRISMA L2D                            | EnMAP L2A                       |
|---------------------------------|---------------------------------------|---------------------------------|
| Scale source                    | HE5 root attributes (per-family)      | Hard-coded class constant       |
| Per-family handling             | Yes (VNIR, SWIR each calibrated separately) | No (uniform across detectors) |
| Sentinel for nodata             | 0 (ambiguous with real zero)          | -32768 (unambiguous int16 min)  |
| Output dtype                    | float32                               | float32                         |
| Uses `numexpr` for performance  | Yes                                   | Yes                             |
| Mask preservation               | Yes (split + reattach)                | Yes (sentinel replacement)      |

The architectural difference is interesting: PRISMA puts the calibration constants in the file (allowing the vendor to update them per scene), while EnMAP puts them in the code (forcing a code release for any change). EnMAP's choice trades flexibility for fewer moving parts.
