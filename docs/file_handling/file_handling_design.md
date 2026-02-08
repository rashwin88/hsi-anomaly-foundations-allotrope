### File helper abstraction

We use the `file_helper` abstract class to create a model for all file handling operations.

```mermaid
classDiagram
    class FileHelper~T, Ab~ {
        <<abstract>>
        file_source_config FileSourceConfig
        file_metadata: T*
        template: Dict*
        access_dataset(path: str) Any
        _construct_metadata_structure()* T
        extract_specific_bands(bands, masking, family, mode)* ndarray
    }

    class FileSourceConfig {
        +path: str
        +source_type: str
    }


    %% Relationships
    FileHelper --> FileSourceConfig : Has-a (Composition)
```

In this specific case the `extract_specific_bands` method goes into the dataset and pulls out bands by their specific index.

### File Handling Design

The `HE5Helper` class implements the file_helper abstraction and is used to handle large HE5 files. Sequence diagrams of individual methods are shown below

```mermaid
sequenceDiagram
    participant U as User
    participant HE5H as HE5Helper
    participant FS as FileSourceConfig

    U->>HE5H: call _construct_metadata_structure()
    activate HE5H
    HE5H->>FS: access file path and config
    FS-->>HE5H: return file path and config
    HE5H->>HE5H: call _get_clean_attrs() for root metadata
    HE5H->>HE5H: visit all dataset paths in file
    loop for each dataset path
        HE5H->>HE5H: call _get_clean_attrs(path)
    end
    HE5H->>HE5H: build metadata objects for each component
    HE5H-->>U: return constructed metadata structure (T)
    deactivate HE5H
```

Extraction of specific bands from the lazily loaded dataset proceeds as follows, not that there is a slight inefficiency that needs correction here. Instead of pulling out specific bands directly, we are forced to pull the entire cube first and then slice out the required bands.

```mermaid
sequenceDiagram
    participant U as User
    participant HE5H as HE5Helper
    participant FS as FileSourceConfig
    participant T as TemplateMapping
    participant H5 as HDF5File

    U->>HE5H: extract_specific_bands(bands, masking_needed, spectral_family, mode)
    activate HE5H
    HE5H->>T: get file path for spectral_family from template mapping
    T-->>HE5H: return path
    HE5H->>H5: access_dataset(path)
    H5-->>HE5H: return raw_cube (full data from path)
    alt mode == "all"
        HE5H->>HE5H: output = raw_cube
    else mode == "specific"
        HE5H->>HE5H: output = slice raw_cube to extract bands by index
    end
    alt masking_needed == true
        HE5H->>HE5H: mask output where values == 0
    end
    HE5H-->>U: return output
    deactivate HE5H
```




### A note on band extraction from files.

There is a subtle but important difference in the way bands are extracted from HE5 and TIF files. In the case of HE5, we pull the entire data cube from the file and then downstream processes can extract whatever band they want. In the case of TIF files, we pull out specific bands directly. Pulling in bands directly will mean a few things. The output of a band extraction is a 3d numpy array. However, since we pull bands out directly in the case of TIF files, the band indexes will get reset and we will have to map them back. This is why we use a band_mapping dictionary in the `BasicBandLevelVisualizationTIF` class. Similar corrections will have to be made whereever TIF files are handled. This is ugly, but necessary till a more comprehensive refactoring is done.

Also note that BIL is the default in the case of HE5 files. However, in the case of TIF files, BSQ is the default. This is why we will have to convert the cube to BIP format for visualization in the `BasicBandLevelVisualizationTIF` class.

*Also note that in the case of TIF files, the bands are indexed starting from 1. This means there is no band 0*


### TIF Helper sequence diagram
```mermaid
sequenceDiagram
    participant U as User
    participant TIFH as TIFHelper
    participant FS as FileSourceConfig
    participant T as TemplateMapping
    participant TIF as TIFFile

    U->>TIFH: extract_specific_bands(bands, masking_needed, spectral_family, mode)
    activate TIFH
    TIFH->>FS: get source_path
    FS-->>TIFH: return .tif file path
    TIFH->>TIF: open source_path using rasterio
    alt mode == "specific"
        TIFH->>TIF: read(bands, masked=masking_needed)
        TIF-->>TIFH: return requested band data (3D np array)
    else mode == "all"
        TIFH->>TIF: read(all bands, masked=masking_needed)
        TIF-->>TIFH: return all band data (3D np array)
    end
    TIFH-->>U: return band data
    deactivate TIFH
```
This sequence diagram illustrates the workflow for band extraction using the `TIFHelper` class. Unlike HE5 files, TIF files allow for direct access to specific bands, resulting in more efficient extraction and memory usage. The user calls `extract_specific_bands`, which retrieves the file path, opens the TIF file via rasterio, and reads either specific bands or all bands depending on the mode. The output is returned as a 3D numpy array suitable for downstream processing or visualization.




