"""
An abstract data transformation class to which all
data transformation functions and classes must adhere to
"""

from abc import ABC, abstractmethod
from typing import Any

from app.models.dataset.transformations import Transformation


class DataTransformer(ABC):
    """
    Builds the data transformation abstract class
    """

    def __init__(self, transformation_category: Transformation):
        """
        Class constructor
        """
        # Set the transformation category.
        self.transformation_category = transformation_category

    @abstractmethod
    def transform(self, input_data: Any, **kwargs) -> Any:
        """
        Performs the actual tranformation
        """
        pass
