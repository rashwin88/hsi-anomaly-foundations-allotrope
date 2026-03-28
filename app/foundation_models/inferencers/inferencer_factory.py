"""
Factory for instantiating the right inferencer from an InferenceConfig.

Maps FoundationModelName → concrete FoundationInferencer class.
Adding a new model = add one entry to _REGISTRY.
"""

from app.abstract_classes.foundation_inferencer import FoundationInferencer
from app.foundation_models.inferencers.spatial_autoencoder_inferencer import (
    SpatialAutoencoderInferencer,
)
from app.foundation_models.inferencers.masked_spatial_autoencoder_inferencer import (
    MaskedSpatialAutoencoderInferencer,
)
from app.foundation_models.inferencers.normalized_masked_autoencoder_inferencer import (
    NormalizedMaskedAutoencoderInferencer,
)
from app.models.training.inference_config import InferenceConfig
from app.models.training.training_config import FoundationModelName

_REGISTRY: dict[FoundationModelName, type[FoundationInferencer]] = {
    FoundationModelName.SPATIAL_AUTOENCODER: SpatialAutoencoderInferencer,
    FoundationModelName.SPATIAL_MASKED_AUTOENCODER_L1_UNNORMALIZED: MaskedSpatialAutoencoderInferencer,
    FoundationModelName.NORMALIZED_MASKED_AUTOENCODER: NormalizedMaskedAutoencoderInferencer,
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
