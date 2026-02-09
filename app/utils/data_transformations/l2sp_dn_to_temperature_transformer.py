"""
Converts the L2SP digital number at the pixel level for
Landsat data LC09 to a temperature scale
"""

from typing import Union

import numpy as np
import numexpr as ne

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation
from app.models.units.surface_temperature import Temperature

# Define constants
SCALING_FACTOR: float = 0.00341802
ADDITIVE_FACTOR: float = 149.0


class Lc09L2spStTransformer(DataTransformer):
    """
    Class for converting LC09 L2SP data to a surface temperature
    """

    def __init__(
        self, transformation_category: Transformation = Transformation.LC09_DN_TO_ST
    ):
        """
        Class constructor
        """
        super().__init__(transformation_category=transformation_category)

    def transform(
        self,
        input_data: Union[np.ndarray, np.ma.MaskedArray],
        unit: Temperature = Temperature.KELVIN,
        **kwargs,
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Performs the actual transformation
        """
        try:
            # We want to do this in a manner that avoids any hidden temporary arrays in the background.
            # And can cause MemoryErrors on large datasets. Using NumExpr solves this problem
            # it is also OS agnostic
            if isinstance(input_data, np.ma.MaskedArray):
                input_dn_buffer = input_data.data
            elif isinstance(input_data, np.ndarray):
                input_dn_buffer = input_data
            # First we pre-allocate
            # This means that there will be exactly one array in memory
            output_data: np.ndarray = np.empty(input_dn_buffer.shape, dtype=np.float32)

            # To prevent scoping issues
            vars_dict = {
                "dn": input_dn_buffer,
                "SF": SCALING_FACTOR,
                "AF": ADDITIVE_FACTOR,
            }

            # We will then execute the computaiton for different Kelvin scales.
            if unit == Temperature.KELVIN:
                ne.evaluate("dn * SF + AF", local_dict=vars_dict, out=output_data)
            elif unit == Temperature.CELSIUS:
                ne.evaluate(
                    "(dn * SF + AF) - 273.15",
                    local_dict=vars_dict,
                    out=output_data,
                )
            elif unit == Temperature.FAHRENHEIT:
                ne.evaluate(
                    "((dn * SF + AF) - 273.15) * 1.8 + 32.0",
                    local_dict=vars_dict,
                    out=output_data,
                )

            # Then depending on the input the output is also formatted and masked accordingly
            if isinstance(input_data, np.ma.MaskedArray):
                return np.ma.masked_array(output_data, mask=input_data.mask)
            elif isinstance(input_data, np.ndarray):
                return output_data
        except Exception as err:
            raise ValueError(
                f"Error in Transform {self.transformation_category} \n {err}"
            ) from err
