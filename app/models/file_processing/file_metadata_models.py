from token import OP
from pydantic import BaseModel, Field
from typing import Any, Optional, Tuple
from typing import Dict, List


class ComponentMetadata(BaseModel):
    """
    Defines the component metadata for the He5 files.
    """

    type: Optional[Any] = Field(default=None, description="The type of the file key")
    shape: Optional[Tuple[int, ...]] = Field(
        default=None, description="The shape of the file ="
    )
    content: Optional[Any] = Field(
        default=None, description="The content of the file key"
    )
    is_scalar: Optional[bool] = Field(
        default=False, description="Whether the file key is a scalar"
    )

    file_attributes: Optional[Dict[str, Any]] = Field(
        default=None, description="The file attributes of the file key"
    )


class FullMetadata(BaseModel):
    """
    Defines the full metadata for the He5 files.
    """

    components: List[str] = Field(
        default=[], description="The components of the He5 files"
    )
    component_metadata: Dict[str, ComponentMetadata] = Field(
        default={}, description="The component metadata for the He5 files"
    )
    root_metadata: ComponentMetadata = Field(
        default=ComponentMetadata(), description="The root metadata for the He5 files"
    )
