"""
Spectral namer: natural or man-made, then the material class (docs/proposals/spectral-namer.md).

    classes.py    v1 class list and the mapping from family-library tags / names
    bank.py       the library resampled to one band list, as tensors with class ids and a train / test split
    model.py      SpectralNamer - a dual encoder on the wavelength-aware stem
    generator.py  planted samples from real tiles, on the GPU
    infer.py      a trained checkpoint on a scene's pixels
    objects.py    per-pixel results -> named objects and their maps (the product's spectral_library_match)
"""
