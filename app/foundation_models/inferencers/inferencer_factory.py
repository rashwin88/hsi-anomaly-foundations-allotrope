"""
Factory for instantiating the right inferencer from an InferenceConfig.

Maps FoundationModelName → concrete FoundationInferencer class.
Adding a new model = add one entry to _REGISTRY.
"""

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.inferencers.spatial_autoencoder_inferencer import (
    SpatialAutoencoderInferencer,
)
from app.models.training.inference_config import InferenceConfig
from app.models.training.training_config import FoundationModelName

_REGISTRY: dict[FoundationModelName, type[FoundationInferencer]] = {
    FoundationModelName.SPATIAL_AUTOENCODER: SpatialAutoencoderInferencer,
}


def get_inferencer(config: InferenceConfig) -> FoundationInferencer:
    """Look up and instantiate the inferencer for the given config."""
    inferencer_cls = _REGISTRY.get(config.foundation_model_name)
    if inferencer_cls is None:
        raise KeyError(
            f"No inferencer registered for '{config.foundation_model_name.value}'. "
            f"Available: {[m.value for m in _REGISTRY]}"
        )
    return inferencer_cls(config)
