### Getting surface temperatures from digital numbers

For the land sat data, the data is processed to the L2SP level, which means that getting to a temperature in kelvin from the digital number is a fairly easy process. We must however be careful of masked pixels and not consider them for temperature conversion.

Looking at the official documentation here: [Link](https://www.usgs.gov/faqs/how-do-i-use-a-scale-factor-landsat-level-2-science-products) the surface temperature in kelvin, given the digital number (DN) in landsat data is simply:

$$
ST = (0.00341802 \times Band10Pixel) + 149.0
$$

This is a very simple linear transformation and there should be a simple function to do this.

