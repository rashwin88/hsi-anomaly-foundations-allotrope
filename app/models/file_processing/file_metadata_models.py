from token import OP
from pydantic import BaseModel, Field
from typing import Any, Optional, Tuple
from typing import Dict, List
import warnings


### Replacing FullMetadata with He5Metadata
def __getattr__(name: str) -> Any:
    if name == "FullMetadata":
        warnings.warn(
            "FullMetadata is deprecated. Please use He5Metadata instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return He5Metadata
    elif name == "ComponentMetadata":
        warnings.warn(
            "ComponentMetadata is deprecated. Please use He5ComponentMetadata instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return He5ComponentMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class He5ComponentMetadata(BaseModel):
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


class He5Metadata(BaseModel):
    """
    Defines the full metadata for the He5 files.
    """

    components: List[str] = Field(
        default=[], description="The components of the He5 files"
    )
    component_metadata: Dict[str, He5ComponentMetadata] = Field(
        default={}, description="The component metadata for the He5 files"
    )
    root_metadata: He5ComponentMetadata = Field(
        default=He5ComponentMetadata(),
        description="The root metadata for the He5 files",
    )


class TIFProperty(BaseModel):
    """
    Defines a single property of the TIFF file
    """

    name: str = Field(description="The name of the property being referenced")
    value: Any = Field(description="The value of the property being referenced")


class TIFMetadata(BaseModel):
    """
    Defines the metadata for the TIFF file
    """

    metadata: Dict[str, TIFProperty] = Field(
        description="The full metadata of the TIFF file"
    )
