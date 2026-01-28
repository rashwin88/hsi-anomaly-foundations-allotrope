## Frequency Identification

One task that is important in hyperspectral analysis is mapping a band in a cube to a specific frquency in the spectrum. There is no metadata in the cube files themselves that can be used to do this. However, root level metadata from the file does have a mapping between band number and the corresponding frequency. We will focus first on the specific metadata slices that will enable us to create this mapping. The technical details of a band mapping helper will be covered in implementation documentation.

## Metadata Slices

In the root metadata there are 6 fields of interest in solving this problem, they are:

In this CW stands for Center Wavelength. and FWHM stands for Full Width at Half Maximum. It is the full width of the sensitivity at half the maximum sensitivity.

|Field|Type|Significance|
|-----|----|------------|
`List_Cw_Swir`|List[float]|The list of center frequencies for the SWIR bands|
`List_Cw_Vnir`|List[float]|The list of center frequencies for the VNIR bands|
`List_Cw_Swir_Flags`|List[int]|The list of flags for the SWIR bands, 1 means that the frequency is valid, 0 means that the frequency is invalid|
`List_Cw_Vnir_Flags`|List[int]|The list of flags for the VNIR bands, 1 means that the frequency is valid, 0 means that the frequency is invalid|
`List_Fwhm_Swir`|List[float]|The list of full width at half maximum for the SWIR bands|
`List_Fwhm_Vnir`|List[float]|The list of full width at half maximum for the VNIR bands|
|-----|----|------------|

We can do some very basic analysis of the data in these fields as follows.

```python    import pprint
    import json

    file_source_config = FileSourceConfig(
        source_path="raw_files/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
    )
    he5_helper = HE5Helper(file_source_config)
    root_attributes = he5_helper.file_metadata.root_metadata.file_attributes

    ## understanding VNIR
    vnir_count = len(root_attributes["List_Cw_Vnir"])
    vnir_non_zero_count = sum(
        1 for x in root_attributes["List_Cw_Vnir_Flags"] if x == 1
    )
    vnir_fwhm_count = len(root_attributes["List_Fwhm_Vnir"])
    vnir_cube_shape = he5_helper.file_metadata.component_metadata[
        "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
    ].shape

    print(f"VNIR Count: {vnir_count}")
    print(f"VNIR Non Zero Count: {vnir_non_zero_count}")
    print(f"VNIR FWHM Count: {vnir_fwhm_count}")
    print(f"VNIR Cube Shape: {vnir_cube_shape}")

    ## understanding SWIR
    swir_count = len(root_attributes["List_Cw_Swir"])
    swir_non_zero_count = sum(
        1 for x in root_attributes["List_Cw_Swir_Flags"] if x == 1
    )
    swir_fwhm_count = len(root_attributes["List_Fwhm_Swir"])
    swir_cube_shape = he5_helper.file_metadata.component_metadata[
        "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
    ].shape

    print(f"SWIR Count: {swir_count}")
    print(f"SWIR Non Zero Count: {swir_non_zero_count}")
    print(f"SWIR FWHM Count: {swir_fwhm_count}")
    print(f"SWIR Cube Shape: {swir_cube_shape}")
```

The output of the above code is as follows:

```shell
VNIR Count: 66
VNIR Non Zero Count: 63
VNIR FWHM Count: 66
VNIR Cube Shape: (1210, 66, 1219)
SWIR Count: 173
SWIR Non Zero Count: 171
SWIR FWHM Count: 173
SWIR Cube Shape: (1210, 173, 1219)
```

Notice a few things here, the `List_Cw_Vnir_FLags` has 66 elements in total and 63 non-zero elements. The indices of the non-zero elements are the bands that are valid. The rest are invalid. The cube has 66 bands which means that it has a band for every wavelength *including* the invalid wavelengths. We can remove them later on. The FWHM list also has 66 elements. For the invalid wavelengths, the FWHM is set to 0.0. 

The `List_Cw_Swir_Flags` has 173 elements in total and 171 non-zero elements. The indices of the non-zero elements are the bands that are valid. The rest are invalid. The cube has 173 bands which means that it has a band for every wavelength *including* the invalid wavelengths. We can remove them later on. The FWHM list also has 173 elements. For the invalid wavelengths, the FWHM is set to 0.0. 

So, by looking at the band number, we can identify the frequency, the FWHM and whether the band is valid or invalid. We will need to construct a helper class that does this.
