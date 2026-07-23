# Appendix A — Notation and Conventions

This appendix collects the abbreviations, symbols, and tensor-layout conventions used throughout Chapter 1. Skim this once and reference it when needed.

---

## A.1 Cube layouts

A hyperspectral or thermal "cube" is a 3-D array of pixel values. Two of the dimensions are spatial (rows and columns); the third is spectral (wavelength bands). The order of those three axes is called the *layout* or *interleave*:

| Layout | Shape       | Bands axis | Used by                                |
|--------|-------------|------------|----------------------------------------|
| **BSQ** (Band-Sequential)        | `(C, H, W)` | first  | Everything except inside the PRISMA builder pre-concat |
| **BIL** (Band-Interleaved-by-Line)| `(H, C, W)` | middle | PRISMA HE5 native layout                |
| **BIP** (Band-Interleaved-by-Pixel)| `(H, W, C)` | last   | Visualization (matplotlib expects channels-last) |

```mermaid
flowchart LR
    A[BSQ C,H,W] -->|swapaxes 0,1| B[BIL H,C,W]
    B -->|swapaxes 1,2| C[BIP H,W,C]
    C -->|moveaxis -1,0| A
```

### Why three layouts exist

Different operations have different cache-locality preferences. For per-pixel spectral analysis (PCA over the spectral axis), you want all of one pixel's spectrum contiguous — that's BIP. For per-band image filtering (FFT a single band), you want a whole band contiguous — that's BSQ. For row-by-row streaming I/O on a tape drive (the historical reason for BIL), you want one row of all bands together — that's BIL.

The codebase standardizes on BSQ everywhere except where the input format forces another layout, and converts on read/write boundaries.

---

## A.2 Digital number and physical quantities

| Symbol  | Name                       | Type / unit                  | Notes                                  |
|---------|----------------------------|------------------------------|----------------------------------------|
| DN      | Digital Number             | int (uint16 or int16)        | What the sensor stores on disk        |
| ρ       | Surface reflectance        | dimensionless, $[0, 1]$       | Physical quantity for hyperspectral   |
| SR      | Surface reflectance (alias for ρ) | same                      | Used in code comments                  |
| ST      | Surface temperature        | K, °C, or °F                  | Physical quantity for thermal         |
| T       | Temperature                | K, °C, or °F                  | Same as ST                            |

Reflectance is defined as the ratio of reflected radiant flux to incident radiant flux at a given wavelength. A perfect white diffuse reflector has $\rho = 1$; a perfect absorber has $\rho = 0$. Real surfaces fall in between.

### Why values >1 occur in practice

Atmospheric correction can over-shoot, producing $\rho > 1$ over snow, clouds, or sun glint. These values are unphysical for diffuse surfaces but are recorded as-is; the validity mask is responsible for flagging them.

---

## A.3 Spectral families

PRISMA, EnMAP, and AVIRIS-NG have two physical detectors covering different wavelength bands. The codebase calls each detector's wavelength range a **spectral family**:

| Family | Name                       | Wavelength range | Notes                          |
|--------|----------------------------|------------------|--------------------------------|
| VNIR   | Visible / Near-Infrared    | ~400–1000 nm     | One detector                   |
| SWIR   | Short-Wave Infrared        | ~1000–2500 nm    | Separate detector              |

The exact boundary depends on the sensor — PRISMA places it at 920 nm in the file, EnMAP uses a detector-index split, AVIRIS-NG codes it as a hard 1000 nm cutoff.

The `SpectralFamily` enum is carried through the entire pipeline as a per-band label. It's used by:

- The PRISMA DN→SR transformer to choose per-family scale factors.
- The composite destriper to run angle detection per family.
- The band filter to apply per-family edge trimming.
- The classical MNF detector to operate per family.

---

## A.4 FWHM

**Full Width at Half Maximum** — the wavelength width of a sensor band at half its peak spectral response. A band centered at 600 nm with FWHM 10 nm responds to roughly 595–605 nm light (half the peak response on each side of center, falling off in a roughly Gaussian shape).

FWHM is carried per-band in the vendable metadata. It is informational — no current pipeline stage uses it for computation — but it is essential for any future cross-sensor reflectance comparison that needs to account for differences in spectral resolution.

---

## A.5 Validity / invalid value masks

A **validity mask** is a binary `(B, H, W)` or `(H, W)` array where 1 means the corresponding voxel is trustworthy and 0 means it is not.

Where the value comes from depends on the sensor:

| Sensor      | Invalid signal                                        |
|-------------|-------------------------------------------------------|
| PRISMA      | Three signals fused: DN ≠ 0, error matrix == 0, vendor band flag == 1 |
| EnMAP       | DN ≠ −32768 (the L2A nodata sentinel)                |
| Landsat     | `MaskedArray` from TIF + adaptive cloud mask + optional QA bit-mask |
| AVIRIS-NG   | Sensor-specific nodata sentinel from .hdr            |
| HotSat      | `UDM == 0` (the Usable Data Mask raster)             |

The validity mask is carried alongside the cube through every pipeline stage. Filled values (Section 12) are *not* re-flagged as valid — the mask stays 0 even when the pixel value has been replaced.

---

## A.6 Mathematical conventions

- Vectors and small constants in inline math: $x, \mu, \sigma$.
- Cubes, matrices: capital letters, e.g., $I, F, V$.
- Element-wise (Hadamard) product: $\odot$.
- Fourier transform: $\mathcal{F}\{I\}$ or $\hat{I}$.
- Statistics over a set: $\mu_j, \sigma_j$ where $j$ is the subset index.

Wavelengths are always in nanometers (nm) unless otherwise stated. Temperatures use whatever unit the surrounding context specifies; the default for downstream models is Celsius.

---

## A.7 File-path conventions in this textbook

Links to source code use the form:

```text
[short_label](../../app/path/to/file.py)
```

The double `../` is because each section file lives at `tech docs/01_data_pipeline/<file>.md`, so reaching `app/` requires going up two directories.

Line numbers, when included, follow the form:

```text
[file.py:NNN](../../app/path/to/file.py)
```

The line number is in the label only — markdown links cannot embed line numbers, but the convention helps you find the spot in the source.

---

## A.8 Common abbreviations

| Abbreviation | Expansion                                                |
|--------------|----------------------------------------------------------|
| ADC          | Analog-to-Digital Converter                              |
| ATBD         | Algorithm Theoretical Basis Document                     |
| BIL/BIP/BSQ  | Cube interleave layouts (Section A.1)                    |
| DLR          | Deutsches Zentrum für Luft- und Raumfahrt (EnMAP operator) |
| DN           | Digital Number                                           |
| EDT          | Exact Euclidean Distance Transform                       |
| ENVI         | Environment for Visualizing Images (file format vendor)  |
| FFT          | Fast Fourier Transform                                   |
| FWHM         | Full Width at Half Maximum                               |
| HE5          | HDF-EOS5 (HDF5 with extension conventions for satellite data) |
| HSI          | Hyperspectral Imaging                                    |
| OPE          | Overlap Patch Embedding (SegFormer input convolution)    |
| PCHIP        | Piecewise Cubic Hermite Interpolating Polynomial         |
| QA           | Quality Assessment                                       |
| SR / ST      | Surface Reflectance / Surface Temperature                |
| STAC         | SpatioTemporal Asset Catalog                             |
| SWIR / VNIR  | Short-Wave Infrared / Visible Near Infrared              |
| UDM          | Usable Data Mask (HotSat)                                |
| USGS         | United States Geological Survey                          |

---

## A.9 Pipeline name list

Quick reference for which transformer corresponds to which conceptual stage:

| Stage                       | Class name                                   | Section |
|-----------------------------|----------------------------------------------|---------|
| DN → reflectance (PRISMA)   | `PrsL2dDnToSurfaceReflectanceTransformer`    | 3       |
| DN → reflectance (EnMAP)    | `EnmapL2aDnToSurfaceReflectanceTransformer`  | 4       |
| DN → temperature (Landsat)  | `Lc09L2spStTransformer`                      | 5       |
| Moment-matching destripe    | `MomentMatchingDestriper`                    | 6       |
| Frequency-domain destripe   | `FrequencyDomainDestriper`                   | 7       |
| Composite destripe          | `CompositeDestriper`                         | 8       |
| Spectral band filter        | `SpectralBandFilter`                         | 9       |
| Spectral gap interpolation  | `SpectralInterpolator`                       | 10      |
| Spectral resampling         | `SpectralResampler`                          | 11      |
| Nearest-valid pixel fill    | `nearest_valid_fill` (function)              | 12      |

Each class inherits from `DataTransformer` and exposes a `transform(input_data, **kwargs)` method (Section 1.4).
