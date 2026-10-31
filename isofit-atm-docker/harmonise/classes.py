"""Per-pixel class from our own reflectance (AC26-NWL option a): no vendor mask exists at deployment."""
import numpy as np

# Snow thresholds are rfl_qc.snow_mask's (NDSI_MIN, NIR_MIN, GREEN_MIN); kept here so harmonise stays standalone.
NDSI_MIN, NIR_MIN, GREEN_MIN = 0.4, 0.11, 0.20
NDWI_MIN, WATER_NIR_MAX = -0.1, 0.15   # AC26-NW: NDWI(560, 860) with a dark-NIR guard


def labels(x, wl):
    """'snow', 'water' or 'land' per row of x (pixels, our bands). Snow wins over water; the rest is land."""
    b = lambda nm: x[:, int(np.argmin(np.abs(np.asarray(wl) - nm)))]
    g, s, n = b(560), b(1600), b(860)
    with np.errstate(invalid="ignore", divide="ignore"):
        snow = ((g - s) / (g + s + 1e-6) > NDSI_MIN) & (n > NIR_MIN) & (g > GREEN_MIN)
        water = ~snow & ((g - n) / (g + n + 1e-6) > NDWI_MIN) & (n < WATER_NIR_MAX)
    return np.where(snow, "snow", np.where(water, "water", "land"))
