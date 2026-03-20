# Building a Vendable: PRISMA vs EnMAP

This document provides a detailed comparison of how `VendableHyperspectralDataset` (PRISMA) and `VendableEnmapHyperspectralDataset` (EnMAP) are constructed. Although both sensors produce hyperspectral imagery and both builders implement the same `DatasetBuilder` ABC, the differences in file format, metadata delivery, cube layout, DN transformation, and quality masking lead to fundamentally different construction pipelines.

---

## 1. High-Level Pipeline Comparison

```
PRISMA:
  Single HE5 File (.he5)
    ├── HE5Helper (h5py)
    ├── Extract SWIR cube + VNIR cube separately (BIL)
    ├── Extract error matrices per family
    ├── Per-family DN→SR transformation (variable scale factors)
    ├── Concatenate SWIR + VNIR along band axis
    ├── Build validity: band_flags × error_pixels × nonzero_pixels
    ├── Convert BIL → BSQ
    └── VendableHyperspectralDataset (239 bands)

EnMAP:
  Scene Folder (13+ files)
    ├── EnmapHelper (rasterio + XML parser)
    ├── Read single merged 224-band cube (already BSQ)
    ├── Read per-band pixel mask from separate TIF
    ├── Uniform DN→SR transformation (gain=0.0001)
    ├── Build validity: nodata_mask × pixel_mask
    ├── Read 5 quality layer TIFs (cloud, cirrus, haze, shadow, snow)
    ├── No cube format conversion needed
    └── VendableEnmapHyperspectralDataset (224 bands)
```

---

## 2. File Format and Source Configuration

### PRISMA — Single HE5 File

PRISMA delivers everything in a single HDF-5 file. The `FileSourceConfig` points directly to it:

```python
config = FileSourceConfig(
    source_path="PRS_L2D_STD_20201214060713_20201214060717_0001.he5"
)
# file_category auto-inferred as FileCategory.HDFS
```

The HE5 file contains a deep hierarchy navigated via h5py:
```
HDFEOS/SWATHS/PRS_L2D_HCO/
├── Data Fields/
│   ├── SWIR_Cube              (H=1210, C=173, W=1219)  ← BIL
│   ├── VNIR_Cube              (H=1210, C=66, W=1219)   ← BIL
│   ├── SWIR_PIXEL_L2_ERR_MATRIX
│   └── VNIR_PIXEL_L2_ERR_MATRIX
└── Geolocation Fields/
    ├── Latitude
    └── Longitude

Root Attributes:
├── List_Cw_Swir[173], List_Cw_Vnir[66]         ← wavelengths
├── List_Cw_Swir_Flags[173], List_Cw_Vnir_Flags[66]  ← validity flags
├── List_Fwhm_Swir[173], List_Fwhm_Vnir[66]     ← FWHM
├── L2ScaleSwirMin/Max, L2ScaleVnirMin/Max       ← scale factors
```

### EnMAP — Folder with Multiple Files

EnMAP delivers a folder containing multiple GeoTIFFs and an XML sidecar:

```python
config = FileSourceConfig(
    source_path="ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"
)
# file_category auto-inferred as FileCategory.ENMAP_FOLDER
```

The folder contents, each prefixed by the scene name:
```
{scene_name}-SPECTRAL_IMAGE.TIF       ← 224 bands, int16, BSQ via rasterio
{scene_name}-QL_PIXELMASK.TIF         ← 224 bands, uint8 (0/1)
{scene_name}-QL_QUALITY_CLOUD.TIF     ← 1 band, uint8
{scene_name}-QL_QUALITY_CIRRUS.TIF    ← 1 band, uint8
{scene_name}-QL_QUALITY_CLOUDSHADOW.TIF
{scene_name}-QL_QUALITY_HAZE.TIF
{scene_name}-QL_QUALITY_SNOW.TIF
{scene_name}-QL_QUALITY_CLASSES.TIF
{scene_name}-METADATA.XML             ← all metadata (wavelengths, quality, bounds)
{scene_name}-HISTORY.XML
```

The `EnmapHelper` resolves individual file paths using the template suffix system:

```python
def _resolve_path(self, component: EnmapFileComponents) -> str:
    suffix = self._template[component].property_name  # e.g., "-SPECTRAL_IMAGE.TIF"
    return os.path.join(self.scene_folder, self.scene_name + suffix)
```

---

## 3. FileHelper and Metadata Extraction

### PRISMA — `HE5Helper`

Metadata is embedded in the HE5 file as root-level HDF5 attributes. The helper traverses the entire hierarchy on initialization:

```python
class HE5Helper(FileHelper[He5Metadata]):
    def __init__(self, file_source_config, template):
        self.raw_structure = h5py.File(file_source_config.source_path, "r")
        self._file_metadata = self._construct_metadata_structure()
```

Wavelengths, FWHMs, validity flags, and scale factors are all read from `root_metadata.file_attributes`:

```python
file_attributes = self.file_helper.file_metadata.root_metadata.file_attributes
swir_wavelengths = file_attributes.get("List_Cw_Swir")     # numpy array of 173 floats
swir_flags = file_attributes.get("List_Cw_Swir_Flags")     # numpy array of 173 ints (0/1)
swir_fwhm = file_attributes.get("List_Fwhm_Swir")          # numpy array of 173 floats
```

Band data is accessed by spectral family — you must specify `SpectralFamily.SWIR` or `SpectralFamily.VNIR` to select which cube to read:

```python
swir_cube = self.file_helper.extract_specific_bands(
    bands=[], spectral_family=SpectralFamily.SWIR, mode="all"
)
# Returns BIL shape: (H=1210, C=173, W=1219)
```

### EnMAP — `EnmapHelper`

Metadata is in a separate XML sidecar file, parsed by `EnmapXmlParser` into a structured Pydantic model:

```python
class EnmapHelper(FileHelper[EnmapMetadata]):
    def __init__(self, file_source_config, template):
        self._xml_parser = EnmapXmlParser(
            self._resolve_path(EnmapFileComponents.METADATA_XML)
        )
        self._file_metadata = self._construct_metadata_structure()
```

The `EnmapMetadata` model is built from XML elements:

```python
metadata.band_characterisation        # List of 224 EnmapBandCharacterisation objects
metadata.detector_boundary            # vnir_expected_channels, swir_expected_channels
metadata.quality_flags                # cloud_cover, haze_cover, etc.
metadata.spatial_info                 # bounding_polygon, width, height, background_value
```

All 224 bands are in a single TIF — no family selection needed to read the cube:

```python
raw_cube = self.file_helper.extract_specific_bands(bands=[], mode="all")
# Returns BSQ shape: (C=224, H=1179, W=1320)
```

EnMAP also provides additional methods not in the ABC:

```python
pixel_mask = self.file_helper.extract_pixel_mask()
# Returns (224, H, W) uint8

cloud_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_CLOUD)
# Returns (H, W) uint8
```

---

## 4. VNIR/SWIR Band Assignment

### PRISMA — Separate Cubes, Sequential Ordering

PRISMA stores SWIR and VNIR as **physically separate cubes** in the HE5 file. The builder processes them sequentially and concatenates:

```python
processing_order = [SpectralFamily.SWIR, SpectralFamily.VNIR]

for family in processing_order:
    bands = self.band_information.get(family)
    unnormalized_cube = self.file_helper.extract_specific_bands(
        bands=[], spectral_family=family, mode="all"
    )
    output_cubes.append(normalized_cube)

# Final cube: SWIR[173] + VNIR[66] = 239 bands
output_cube = np.concatenate(output_cubes, axis=1)
```

Band metadata (wavelengths, flags, FWHM) comes from separate root attributes per family:
- `List_Cw_Swir[173]` + `List_Cw_Vnir[66]`
- `List_Cw_Swir_Flags[173]` + `List_Cw_Vnir_Flags[66]`

The resulting `spectral_family_order` is a clean split: 173 SWIR entries followed by 66 VNIR entries.

### EnMAP — Single Merged Cube, Interleaved Detectors

EnMAP stores all 224 bands in a **single GeoTIFF**. The bands are ordered by channel number, but the VNIR and SWIR detectors are **interleaved** in the overlap region (channels 79-101):

```
VNIR channels: 1-78, 81, 83, 84, 86, 88, 90, 92, 93, 95, 97, 99, 101
SWIR channels: 80, 82, 85, 87, 89, 91, 94, 96, 98, 100, 102-224
```

The builder determines each band's family by checking the XML `expectedChannelsList`:

```python
vnir_channels = set(metadata.detector_boundary.vnir_expected_channels)

for band_char in metadata.band_characterisation:
    if band_char.band_number in vnir_channels:
        spectral_family_order.append(SpectralFamily.VNIR)
    else:
        spectral_family_order.append(SpectralFamily.SWIR)
```

The resulting `spectral_family_order` has interleaved VNIR/SWIR entries in the 79-101 range — not a clean two-block split like PRISMA.

---

## 5. DN-to-Surface-Reflectance Transformation

### PRISMA — Per-Family Variable Scale Factors

PRISMA L2D stores raw DN values as uint16 in [0, 65535]. The conversion requires **per-family scale factors** read from HE5 root metadata:

```python
# Formula: SR = DN × (max - min) / 65535 + min
vnir_scale = (vnir_max - vnir_min) / 65535  # different for each family
swir_scale = (swir_max - swir_min) / 65535

# Build per-band scaling arrays for numexpr broadcasting
scaling_factors = np.empty((num_bands, 1, 1), dtype=np.float32)
additive_factors = np.empty((num_bands, 1, 1), dtype=np.float32)
for i, family in enumerate(band_mapping):
    if family == SpectralFamily.SWIR:
        scaling_factors[i] = swir_scale
        additive_factors[i] = swir_add
    elif family == SpectralFamily.VNIR:
        scaling_factors[i] = vnir_scale
        additive_factors[i] = vnir_add

ne.evaluate("dn * SF + AF", local_dict=vars_dict, out=output_data)
```

The transformer also handles cube format conversion (BIL → BSQ → BIL) internally, and manages masked arrays through the transformation.

### EnMAP — Uniform Gain Across All Bands

EnMAP L2A has already been atmospherically corrected. The DN values are int16 with a **uniform gain of 0.0001 and offset of 0** across all 224 bands:

```python
# Formula: SR = DN × 0.0001
class EnmapL2aDnToSurfaceReflectanceTransformer(DataTransformer):
    GAIN = 0.0001
    OFFSET = 0.0

    def transform(self, input_data, nodata_value=-32768, **kwargs):
        output = np.empty(input_array.shape, dtype=np.float32)
        gain = np.float32(self.GAIN)
        offset = np.float32(self.OFFSET)
        ne.evaluate("input_array * gain + offset",
                     local_dict={"input_array": input_array, "gain": gain, "offset": offset},
                     out=output)
        # Zero out nodata pixels
        output[input_array == nodata_value] = 0.0
        return output
```

No per-family lookup, no cube format conversion, no masked array handling needed. The gain and offset values can be verified from the XML metadata where every band reports `GainOfBand=0.0001` and `OffsetOfBand=0`.

---

## 6. Validity Mask Construction

### PRISMA — Three-Signal Intersection

PRISMA builds validity from **three independent sources**, all derived from the HE5 file:

```python
# Signal 1: Non-zero pixel values (DN != 0 means valid)
invalid_value_mask = (unnormalized_cube != 0.0).astype(np.int8)

# Signal 2: Error matrices (error == 0 means valid)
error_pixels = self.file_helper.extract_error_matrices(
    bands=[], spectral_family=family, mode="all"
)
error_pixel_mask = (error_pixels == 0.0).astype(np.int8)

# Signal 3: Per-band validity flags broadcast to cube shape
band_validity = np.asarray(band_validity_by_position, dtype=np.uint8)  # from metadata
valid_band_cube = np.broadcast_to(band_validity[None, :, None], output_cube.shape)

# Combine: all three must agree for a pixel to be valid
overall_validity = valid_band_cube * error_pixel_mask * invalid_value_mask
```

This three-way multiplication means a pixel is only valid if:
1. Its raw DN is non-zero
2. Its error matrix entry is zero (no processing error)
3. Its band is flagged as valid in the metadata

### EnMAP — Two-Signal Intersection + Separate Quality Layers

EnMAP builds core validity from **two sources** — the nodata sentinel and a dedicated pixel mask TIF:

```python
# Signal 1: Nodata mask (DN != -32768 means valid)
nodata_mask = (raw_cube != metadata.spatial_info.background_value).astype(np.int8)

# Signal 2: Per-band pixel mask from QL_PIXELMASK.TIF (224 bands, 0/1)
pixel_mask = self.file_helper.extract_pixel_mask()

# Combine: both must agree
overall_validity = nodata_mask * pixel_mask.astype(np.int8)
```

Quality information (cloud, cirrus, haze, cloud shadow, snow) is stored as **separate optional fields** rather than folded into the validity cube. This gives downstream consumers the flexibility to decide which quality signals to apply:

```python
cloud_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_CLOUD)
cirrus_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_CIRRUS)
haze_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_HAZE)
cloud_shadow_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_CLOUDSHADOW)
snow_mask = self.file_helper.extract_quality_mask(EnmapFileComponents.QUALITY_SNOW)
```

---

## 7. Cube Format and Final Conversion

### PRISMA — BIL Native, Requires Conversion

PRISMA stores cubes in **BIL (Band Interleaved by Line)** format: `(H, C, W)`. The vendable pipeline standardizes everything to BSQ, so a conversion step is required at the end:

```python
@property
def default_cube_representation(self) -> CubeRepresentation:
    return CubeRepresentation.BIL

# In vend_dataset():
return VendableHyperspectralDataset(
    normalized_hyperspectral_cube=self.cube_reshaper.convert_cube(
        cube=output_cube,
        from_format=self.default_cube_representation,  # BIL
        to_format=CubeRepresentation.BSQ,              # → BSQ
    ),
    validity_cube=self.cube_reshaper.convert_cube(
        cube=overall_validity_mask,
        from_format=self.default_cube_representation,
        to_format=CubeRepresentation.BSQ,
    ),
    ...
)
```

`ImageCubeOperations.convert_cube()` performs the permutation via `torch.permute()`: BIL `(H, C, W)` → BSQ `(C, H, W)` is a `(1, 0, 2)` permutation.

### EnMAP — BSQ Native, No Conversion Needed

EnMAP is read via rasterio which returns data in **BSQ (Band Sequential)** format: `(C, H, W)`. No conversion is needed:

```python
@property
def default_cube_representation(self) -> CubeRepresentation:
    return CubeRepresentation.BSQ

# In vend_dataset():
return VendableEnmapHyperspectralDataset(
    normalized_hyperspectral_cube=sr_cube,    # Already BSQ, no conversion
    validity_cube=overall_validity,            # Already BSQ, no conversion
    ...
)
```

This eliminates the `ImageCubeOperations` dependency entirely — the `EnmapDatasetBuilder` does not instantiate `cube_reshaper` at all.

---

## 8. Vendable Output Model Comparison

### `VendableHyperspectralDataset` (PRISMA)

```python
class VendableHyperspectralDataset(BaseModel):
    normalized_hyperspectral_cube: np.ndarray   # (239, H, W) float32, BSQ
    validity_cube: np.ndarray                   # (239, H, W) int8
    spectral_family_order: List[SpectralFamily]  # [SWIR]*173 + [VNIR]*66
    band_cw_order: List[float]                   # 239 wavelengths (nm)
    band_fwhm_order: Optional[List[float]]       # optional
    band_validity_by_position: List[int]          # per-band 0/1 from metadata
```

### `VendableEnmapHyperspectralDataset` (EnMAP)

```python
class VendableEnmapHyperspectralDataset(BaseModel):
    normalized_hyperspectral_cube: np.ndarray   # (224, H, W) float32, BSQ
    validity_cube: np.ndarray                   # (224, H, W) int8
    spectral_family_order: List[SpectralFamily]  # interleaved VNIR/SWIR in overlap region
    band_cw_order: List[float]                   # 224 wavelengths (nm)
    band_fwhm_order: List[float]                 # 224 FWHMs
    band_validity_by_position: List[int]          # all 1s for L2A

    # Quality layers (EnMAP-specific)
    cloud_mask: Optional[np.ndarray]              # (H, W) uint8
    cirrus_mask: Optional[np.ndarray]             # (H, W) uint8
    haze_mask: Optional[np.ndarray]               # (H, W) uint8
    cloud_shadow_mask: Optional[np.ndarray]       # (H, W) uint8
    snow_mask: Optional[np.ndarray]               # (H, W) uint8

    # Detector boundary (EnMAP-specific)
    vnir_channel_indices: List[int]               # 91 channel numbers
    swir_channel_indices: List[int]               # 133 channel numbers
```

Key differences:
- EnMAP adds 5 quality mask fields and 2 channel index fields
- EnMAP `band_fwhm_order` is non-optional (always populated from XML)
- EnMAP `band_validity_by_position` is always all 1s (L2A = all bands valid)
- PRISMA `band_validity_by_position` can have 0s (some bands flagged invalid in metadata)

---

## 9. Summary Table

| Aspect | PRISMA | EnMAP |
|--------|--------|-------|
| **File format** | Single HE5 (HDF-5) | Folder: GeoTIFFs + XML |
| **Helper class** | `HE5Helper` (h5py) | `EnmapHelper` (rasterio + XML) |
| **Metadata source** | Embedded HE5 root attributes | XML sidecar file |
| **Band cubes** | Separate SWIR + VNIR, concatenated | Single merged 224-band TIF |
| **Native cube format** | BIL `(H, C, W)` | BSQ `(C, H, W)` |
| **Format conversion** | BIL → BSQ via `ImageCubeOperations` | None needed |
| **Total bands** | 239 (173 SWIR + 66 VNIR) | 224 (91 VNIR + 133 SWIR) |
| **VNIR/SWIR layout** | Clean two-block split | Interleaved in overlap region |
| **DN data type** | uint16 [0, 65535] | int16 [-32768, 32767] |
| **Nodata sentinel** | 0 (zero means invalid) | -32768 |
| **DN formula** | `SR = DN × (max-min)/65535 + min` | `SR = DN × 0.0001` |
| **Scale factors** | Per-family (SWIR/VNIR have different scales) | Uniform (0.0001 for all bands) |
| **Error matrices** | Available (per-band pixel error) | Not applicable |
| **Per-band validity** | From metadata flags (some bands invalid) | All bands valid in L2A |
| **Quality layers** | None (no cloud/haze/snow masks) | 5 separate TIF masks |
| **Validity formula** | `band_flag × error_matrix × nonzero_dn` | `nodata_mask × pixel_mask` |
| **CRS** | Sensor-specific (from lat/lon arrays) | EPSG:4326 (geographic) |
| **Vendable class** | `VendableHyperspectralDataset` | `VendableEnmapHyperspectralDataset` |
