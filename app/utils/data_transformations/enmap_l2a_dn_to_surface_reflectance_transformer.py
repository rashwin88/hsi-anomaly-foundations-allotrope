"""
Performs transformation of EnMAP L2A data from DN to surface reflectance.
For L2A products, the transformation is: SR = DN * Gain + Offset
where Gain = 0.0001 and Offset = 0 (uniform across all 224 bands).
"""

import logging
from typing import Union

import numpy as np
import numexpr as ne

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation


logger = logging.getLogger("EnmapL2aDnToSurfaceReflectanceTransformer")
logger.setLevel(logging.INFO)


class EnmapL2aDnToSurfaceReflectanceTransformer(DataTransformer):
    """
    Converts EnMAP L2A int16 DN values to surface reflectance.
    """

    GAIN: float = 0.0001
    OFFSET: float = 0.0

    def __init__(self):
        super().__init__(
            transformation_category=Transformation.ENMAP_L2A_DN_TO_SR
        )

    def transform(
        self,
        input_data: Union[np.ndarray, np.ma.MaskedArray],
        nodata_value: int = -32768,
        **kwargs,
    ) -> np.ndarray:
        """
        Transforms DN cube to surface reflectance.

        Args:
            input_data: Raw DN cube (C, H, W) as int16.
            nodata_value: Value indicating nodata pixels (default: -32768).

        Returns:
            Surface reflectance cube as float32 with nodata pixels set to 0.0.
        """
        input_array = (
            input_data.data
            if isinstance(input_data, np.ma.MaskedArray)
            else input_data
        )

        output = np.empty(input_array.shape, dtype=np.float32)
        gain = np.float32(self.GAIN)
        offset = np.float32(self.OFFSET)

        ne.evaluate(
            "input_array * gain + offset", local_dict={"input_array": input_array, "gain": gain, "offset": offset}, out=output
        )

        # Zero out nodata pixels
        nodata_mask = input_array == nodata_value
        output[nodata_mask] = 0.0

        logger.info("Transformed cube shape: %s, dtype: %s", output.shape, output.dtype)
        return output
