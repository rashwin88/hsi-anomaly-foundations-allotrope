"""PRISMA L2D georef from the product's own UTM grid (app/georef/from_scene_file.py prisma_utm_grid)."""

import numpy as np

from app.georef.from_scene_file import prisma_utm_grid

# AIGC Set-1 (PRS_L2D_STD_20201214060713_20201214060717_0001.he5) root attributes, as float32 like the file,
# and its ground-truth raster's grid (research/experiments/gt-had-prisma/set1/metrics.json).
SET1 = {"Epsg_Code": np.uint32(32642),
        "Product_ULcorner_easting": np.float32(545022.75), "Product_ULcorner_northing": np.float32(2501422.5),
        "Product_URcorner_easting": np.float32(580872.5), "Product_LLcorner_northing": np.float32(2465872.5)}
GT = (30.0, 0.0, 545007.75, 0.0, -30.0, 2501437.5)


def test_set1_attributes_give_the_ground_truth_grid():
    transform, epsg = prisma_utm_grid(SET1, (1186, 1196))
    assert epsg == 32642
    assert np.allclose(tuple(transform)[:6], GT, atol=1e-6)


def test_missing_attributes_fall_back():
    assert prisma_utm_grid({k: v for k, v in SET1.items() if k != "Epsg_Code"}, (1186, 1196)) is None
    assert prisma_utm_grid(SET1, (1, 1196)) is None
