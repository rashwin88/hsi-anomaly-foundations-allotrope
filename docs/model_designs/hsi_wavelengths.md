## Making wavelength look up easy

We know that the he5 files provided contain very useful information about the wavelengths and other lookups. The idea is to encapsulate all the wavelength  information behind a class so thayt given a he5 file, all sorts of operations on the wavelength can be performed.

### Design for the wavelength datamodel

We know that there is a correspondence between a wavelength (or a central wavelength) and a band id or index in the hyperspectral cube. In addition, there is also a FWHM parameter which is important to capture. So, we can define a `Band` with the following properties.

1. Central Wavelength
2. FWHM
3, Band Index - what is the index of the band in the respective cube.

