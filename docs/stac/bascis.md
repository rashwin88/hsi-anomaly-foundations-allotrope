## Basics of STAC

Stands for Spatio-Temporal Asset Catalog. It is simple and extensible and would be a good candidate for some of the abstraction problems we are facing without having to create hundreds of new Enums and dataclasses. 

It is not a fully fledged metadata standard, which is perfect for our usecase.

## The Specification

The specification itself consists of 4 parts:
1. Catalogs
2. Collections
3. Items
4. API

**Catalogs** are basic groupings for items, catalogs and collections.
**Collection** is a catalog with rich metadata. 

### What does this mean? 
A catalog is like a folder or a directory whose only job is to group things together so that we can browse through them. A collection is like a product or a dataset. It refers to a coherent set of data sharing common properties. 

A collection is a type of catalog but has strict requirements added to it. It makes data searchable and discoverable. On the other hand, we can only crawl through a catalog. 

### Required fields in a catalog
1. `stac_verison (string) `implemented by the catalog.
2. `id`- an identifier for the catalog.
3. `description`for the catalog
4. `links`- A list of references to other documents.

Note here that we talk only about references to other documents. The document in itself has nothing to identify it with the catalog. This is a weak link where information is stored only in the catalog and not in the documents that the catalog refers to.

In addition to these fields, a collection has some other required fields (remember that a collection is a catalog with some additional strict requirements).

1. `license`of the collection
2. ` extent`- spatial and temporal extent.


## Items and Assets
Items are single scenes or a set of datafiles for a specific location at a specific date and time. Each item has an asset which is a single data file for download. Each asset can have its own metadata. Properties of an item are basically just key value pairs. There are some *core* properties and *extensions*. Asset has a href which is a link to the asset object.

Items and assets are of interest to the specific problem we are solving here. 

-----

Here is an example of an Item

```json
{
    "stac_version": "1.0.0-beta.2",
    "stac_extension": [],
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "type" : "Feature", // Stack Items are always features GeoJSON
    "geometry" : "", // standard geometry
    "bounding_box" : [],
    "properties" : {
        "datetime" : "" // Must contain this
    },
    "links" : [],
    "assets" : {}
}

```



