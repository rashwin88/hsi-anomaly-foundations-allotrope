from token import OP
from pydantic import BaseModel, Field
from typing import Any, Optional, Tuple


class FileMetadata(BaseModel):

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
