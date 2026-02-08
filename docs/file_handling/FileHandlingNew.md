## Separating responsibilities in File Handling

In this applicaiton, a conscious decision has been made to seperate lowl level operations that involve direct interaction with the files from actually understanding what the file represents and structuring that information. The files that we deal with come from specific satellites (platforms) and are packaged as products that are processed in a specific manner. It is not necessary to create seperate classes and data structures to understand these files. STAC is a rather convenient framework and all the nuances of parsing file names and understanding what the file represents. So, effectively we will use custom classes and methods to deal with the nuances of extracting data from the files, but will use the STAC framework to represent the dataset with some structure.

<p align="center">
  <img src="assets/modified-file-handling.png" width="500" alt="File Handling Flow">
</p>

## The STAC creation process

To create a stac item we need to be able to extract specific properties from the file. Most of these properties are available from the name of the file itself, some things such as the bounding box and geometry are extractable from a thin reading of the file itself. The process is shown below

```mermaid
sequenceDiagram
    participant U as User
    participant SC as StacCreator
    participant FNP as FileNameParser
    participant BBL as BoundingBoxLib
    participant STAC as STAC Item

    U->>SC: __init__(file_path)
    activate SC
    SC->>SC: file_name and file_id extraction
    SC->>SC: Determine media_type from file_name
    SC->>FNP: parse(file_name)
    FNP-->>SC: metadata
    alt platform == Prisma
        SC->>BBL: get_prisma_bounding_box(file_path)
        BBL-->>SC: bounding_box
        SC->>SC: asset_role = hyperspectral
    else platform == landsat-9
        SC->>BBL: get_landsat_bounding_box(file_path)
        BBL-->>SC: bounding_box
        SC->>SC: asset_role = thermal
    end
    SC->>SC: _build_geojson_geometry()

    U->>SC: build_stack()
    SC->>SC: Build item_props from metadata
    SC->>STAC: Create Item with geom, bbox, props
    SC->>STAC: Create Asset (with href, media_type, roles)
    SC->>STAC: Add Asset as "primary_input_datacube"
    SC-->>U: return STAC Item
    deactivate SC
```
This diagram shows how the `StacCreator` class takes a `file_path`, processes the file to extract metadata and geometry, and produces a STAC Item and associated Asset using helper functions, neatly capturing the steps from construction to item creation.



