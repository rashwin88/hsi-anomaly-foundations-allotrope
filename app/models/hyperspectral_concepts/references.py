"""
Different Reference Types
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
