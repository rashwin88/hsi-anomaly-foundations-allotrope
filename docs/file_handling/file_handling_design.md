### File Handling Design

This project will involve a lot of he5 file handling and it is important to have a consistent system of managing these files.
What we are dealing with is a specific flavor of the hierarchical data format 5 file - HDF-EOS5. This is a standard format used in satellite data. The file is self describing and usually contains metadata tags.

Think of the 'file' as a tree structure. The root node is the file itself. The leaves are the datasets. The branches are the groups. We need a simple interface to help access the data in the file.

In this discussion we will seperate **downloading** the file from *processing* it. This implies that for file processing to take place the file object is already downloaded and available in a local path.

The local path is stored in the `FileSourceConfig` model where the `source_path` field is a string.

The `he5_helper` class is used to help process the file. It is initialized with a `FileSourceConfig` object and provides a simple interface to help access the data in the file.

The general flow of the file handling is as follows:

<p align="center">
  <img src="assets/he5_helper.png" width="700" alt="File Handling Flow">
</p>

We pass a path of the he5 file to the he5 helper class. This class will then pull out the metadata at the file and group level and contain methods to access the data at the dataset level.
