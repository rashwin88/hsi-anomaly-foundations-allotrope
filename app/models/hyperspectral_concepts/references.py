"""
How a template says "this component lives HERE inside the file".

The building block of the template system (app/templates/). A sensor template is
a dict of {logical component -> ReferenceDefinition}, and each definition names
one of three retrieval strategies, because the three container formats hide
their contents in fundamentally different places:

    FILE_REFERENCE              an internal hierarchical path, e.g. the HDF-EOS
                                dataset "HDFEOS/SWATHS/.../SWIR_Cube"
    ROOT_METADATA_FIELD         a root-level attribute, e.g. PRISMA's
                                "List_Cw_Swir" - a different h5py call entirely
    DIRECT_PROPERTY_DEFINITION  a property on the opened dataset object, e.g.
                                rasterio's `crs` or `bounds`

The model validator enforces that the field matching the chosen type is actually
populated, so a malformed template fails at import rather than at read time -
when it would surface as a confusing None deep inside a helper.

One wart: the EnMAP template overloads DIRECT_PROPERTY_DEFINITION to carry
FILENAME SUFFIXES ("-SPECTRAL_IMAGE.TIF") rather than object attributes.
Semantically that is closer to a file reference, but the validator would reject
file_name on that type. Worth knowing before you trust property_name to mean
"attribute".
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class ReferenceType(Enum):
    """
    The different types of references
    """

    ROOT_METADATA_FIELD = "root_metadata"
    FILE_REFERENCE = "file_reference"
    DIRECT_PROPERTY_DEFINITION = "direct_property_definition"


class ReferenceDefinition(BaseModel):
    """
    How a reference is defined.
    """

    description: str = Field(
        ..., description="The description of the Reference Definition"
    )

    reference_type: ReferenceType = Field(
        ..., description="The type of reference that is being defined"
    )

    file_name: Optional[str] = Field(
        default=None,
        description="The name of the file in the hierarchical file definition, if the reference is appropriate",
    )

    root_metadata_field_name: Optional[str] = Field(
        default=None, description="The name of the root metadata field"
    )

    property_name: Optional[str] = Field(
        default=None,
        description="The name of the property that must be extracted in the case of TIF files .etc",
    )

    @model_validator(mode="after")
    def check_reference_requirements(self) -> "ReferenceDefinition":
        """
        Checks whether the reference definition is valid.
        """
        if self.reference_type == ReferenceType.FILE_REFERENCE:
            if not self.file_name:
                raise ValueError(f"{ReferenceType.FILE_REFERENCE} needs a file name.")
        elif self.reference_type == ReferenceType.ROOT_METADATA_FIELD:
            if not self.root_metadata_field_name:
                raise ValueError(
                    f"{ReferenceType.ROOT_METADATA_FIELD} needs a field name."
                )
        elif self.reference_type == ReferenceType.DIRECT_PROPERTY_DEFINITION:
            if not self.property_name:
                raise ValueError(
                    f"{ReferenceType.DIRECT_PROPERTY_DEFINITION} needs a property name."
                )
        return self
