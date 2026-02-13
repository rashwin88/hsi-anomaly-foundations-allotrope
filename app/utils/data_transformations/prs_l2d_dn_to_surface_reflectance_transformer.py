"""
Performs transformation of Prisma Data from DN to surface reflectance
"""

import logging
from typing import Union, List, Optional

import numpy as np
import numexpr as ne

from app.abstract_classes.data_transformer import DataTransformer
from app.models.dataset.transformations import Transformation
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily
from app.models.file_processing.file_metadata_models import He5Metadata
from app.models.images.cube_representation import CubeRepresentation
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.models.hyperspectral_concepts.file_components import (
    HyperspectralFileComponents,
)
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier


logger = logging.getLogger("PrsL2dDnToSurfaceReflectanceTransformer")
logger.setLevel(logging.INFO)

PRISMA_DIV_FACTOR = 65535


class PrsL2dDnToSurfaceReflectanceTransformer(DataTransformer):
    """
    Class that converts L2D PRS data in SWIR and VNIR cubes to a surface reflectance
    """

    def __init__(
        self, transformation_category: Transformation = Transformation.PRS_L2D_DN_TO_SR
    ):
        """
        Class constructor
        """
        super().__init__(transformation_category=transformation_category)
        self.cube_operations = ImageCubeOperations()

    def transform(
        self,
        input_data: Union[np.ndarray, np.ma.MaskedArray],
        cube_representation: CubeRepresentation,
        band_mapping: List[SpectralFamily],
        file_metadata: Optional[He5Metadata] = None,
        masking_indicator: float = 0.0,
        **kwargs
    ) -> Union[np.ndarray, np.ma.MaskedArray]:
        """
        Transfroms band level hyperspectral data from digital numbers to
        surface reflectance values. Made as generic as possible to avoid
        any linkages with concepts deeply embedded in upstream layers
        """

        # Perform a check to see if the transformation can be supported given the product
        if self.transformation_category not in [Transformation.PRS_L2D_DN_TO_SR]:
            raise NotImplementedError(
                "Cannot support this transformation type at present"
            )

        # From a CPU perspective casting everything in BSQ seems to be the
        # Best fit and hence we will do that
        input_representation: CubeRepresentation = cube_representation

        # The process of conversion will be as follows.
        # We will operate on unmasked numpy arrays only
        input_array: np.ndarray = None
        input_mask: np.ndarray = None
        masked_input_flag: bool = False

        if isinstance(input_data, np.ma.MaskedArray):
            input_array = input_data.data
            masked_input_flag = True
            input_mask = input_data.mask
        elif isinstance(input_data, np.ndarray):
            input_array = input_data
        else:
            raise TypeError("Format not supported")

        # Shape things up
        if input_representation == CubeRepresentation.BSQ:
            pass
        else:
            input_array = self.cube_operations.convert_cube(
                cube=input_array,
                from_format=input_representation,
                to_format=CubeRepresentation.BSQ,
                output_form="numpy",
            )
            if masked_input_flag:
                input_mask = self.cube_operations.convert_cube(
                    cube=input_mask,
                    from_format=input_representation,
                    to_format=CubeRepresentation.BSQ,
                    output_form="numpy",
                )
                input_mask
        # Now we have the input data in the right shape and other the mask if applicable.
        # One safe bet is to conver the input data array to 0 where ever there is a mask
        if masked_input_flag:
            input_array[input_mask] = masking_indicator

        # First we pre-allocate
        # This means that there will be exactly one array in memory
        output_data: np.ndarray = np.empty(input_array.shape, dtype=np.float32)
        logger.info("Output array shape: %s", output_data.shape)

        # Now this is where we need to be a bit careful
        # We will have SWIR bands and we will have VNIR bands too.
        # Need to esnure that the scale factors for these bands are applied properly
        # Or numexpr will have a problem.

        # To make the conversion, based on spectral family
        # We will need an additive factor and a scaling factor
        # Additive factor is the min scale
        # Scaling factor is the max - min divided by the appropriate DIV factor
        # First construct empty arrays
        # Remember that shapes are super important or broadcasting will fail
        scaling_factors = np.empty((input_array.shape[0], 1, 1), dtype=np.float32)
        additive_factors = np.empty((input_array.shape[0], 1, 1), dtype=np.float32)
        if self.transformation_category == Transformation.PRS_L2D_DN_TO_SR:
            # Get the appropriate template
            template = TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL)
            # Having got the template we can store the different factors
            vnir_max = file_metadata.root_metadata.file_attributes.get(
                template.get(
                    HyperspectralFileComponents.L2_SCALE_MAX_VNIR
                ).root_metadata_field_name
            )
            vnir_min = file_metadata.root_metadata.file_attributes.get(
                template.get(
                    HyperspectralFileComponents.L2_SCALE_MIN_VNIR
                ).root_metadata_field_name
            )
            swir_max = file_metadata.root_metadata.file_attributes.get(
                template.get(
                    HyperspectralFileComponents.L2_SCALE_MAX_SWIR
                ).root_metadata_field_name
            )
            swir_min = file_metadata.root_metadata.file_attributes.get(
                template.get(
                    HyperspectralFileComponents.L2_SCALE_MIN_SWIR
                ).root_metadata_field_name
            )
            # Then we can compute the swir and vnir additive and scales
            vnir_add: float = vnir_min
            vnir_scale: float = (vnir_max - vnir_min) / PRISMA_DIV_FACTOR
            swir_add: float = swir_min
            swir_scale: float = (swir_max - swir_min) / PRISMA_DIV_FACTOR

            # Now these are obtained, we can loop and populate the respective factors
            for i, spectral_family in enumerate(band_mapping):
                if spectral_family == SpectralFamily.SWIR:
                    scaling_factors[i] = swir_scale
                    additive_factors[i] = swir_add
                elif spectral_family == SpectralFamily.VNIR:
                    scaling_factors[i] = vnir_scale
                    additive_factors[i] = vnir_add
        else:
            # Nothing to do here
            # We would have errored out earlier
            pass

        # Now we can actually make the numerical computation
        vars_dict = {"dn": input_array, "SF": scaling_factors, "AF": additive_factors}
        ne.evaluate("dn * SF + AF", local_dict=vars_dict, out=output_data)

        # Now we can do the post processing
        if masked_input_flag:
            output_data = np.ma.masked_where(input_mask, output_data)

        if input_representation == CubeRepresentation.BSQ:
            return output_data
        else:
            return self.cube_operations.convert_cube(
                cube=output_data,
                from_format=CubeRepresentation.BSQ,
                to_format=input_representation,
                output_form="numpy",
            )
