"""
An abstract class for an ML / Statistical Model
"""

from typing import Any
from abc import ABC
from app.models.base_models.base_model import BaseModel


class MlModel(ABC):
    """
    Abstractions for all ML and statistical models
    """

    def __init__(self, base_model: BaseModel):
        """
        class constructor
        """
        self.base_model = base_model

    def configure(self, **kwargs):
        """
        Configure the model
        """
        pass

    def train(self, **kwargs):
        """
        Train the model
        """
        pass

    def predict(self, **kwargs) -> Any:
        """
        Predict using the model.
        Being quite permissive in the case of return signatures.
        """
        pass
