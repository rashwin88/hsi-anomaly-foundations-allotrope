"""
Factory for instantiating the right trainer from a TrainingConfig.

Maps FoundationModelName → concrete FoundationTrainer class.
Adding a new model = add one entry to _REGISTRY.
"""

from app.abstract_classes.foundation_trainer import FoundationTrainer
from app.foundation_models.trainers.spatial_autoencoder_trainer import (
    SpatialAutoencoderTrainer,
)
from app.foundation_models.trainers.spatial_masked_autoencoder_trainer import (
    SpatialMaskedAutoencoderTrainer,
)
from app.foundation_models.trainers.spatial_masked_autoencoder_trainer_l1_loss import (
    SpatialMaskedAutoencoderTrainerL1Loss,
)
from app.foundation_models.trainers.spatial_masked_autoencoder_trainer_l1_loss_unnormalized import (
    UnNormalizedSpatialMaskedAutoencoderTrainerL1Loss
)
from app.foundation_models.trainers.normalized_masked_autoencoder_trainer import (
    NormalizedMaskedAutoencoderTrainer,
)
from app.foundation_models.trainers.segformer_mae_trainer import (
    SegFormerMAETrainer,
)
from app.foundation_models.trainers.hyperspectral_segformer_mae_trainer import (
    HyperspectralSegFormerMAETrainer,
)

from app.models.training.training_config import (
    FoundationModelName,
    TrainingConfig,
)

_REGISTRY: dict[FoundationModelName, type[FoundationTrainer]] = {
    FoundationModelName.SPATIAL_AUTOENCODER: SpatialAutoencoderTrainer,
    FoundationModelName.SPATIAL_MASKED_AUTOENCODER: SpatialMaskedAutoencoderTrainer,
    FoundationModelName.SPATIAL_MASKED_AUTOENCODER_L1: SpatialMaskedAutoencoderTrainerL1Loss,
    FoundationModelName.SPATIAL_MASKED_AUTOENCODER_L1_UNNORMALIZED: UnNormalizedSpatialMaskedAutoencoderTrainerL1Loss,
    FoundationModelName.NORMALIZED_MASKED_AUTOENCODER: NormalizedMaskedAutoencoderTrainer,
    FoundationModelName.SEGFORMER_MAE: SegFormerMAETrainer,
    FoundationModelName.HYPERSPECTRAL_SEGFORMER_MAE: HyperspectralSegFormerMAETrainer,
}


def get_trainer(config: TrainingConfig) -> FoundationTrainer:
    """Look up and instantiate the trainer for the given config."""
    trainer_cls = _REGISTRY.get(config.foundation_model_name)
    if trainer_cls is None:
        raise KeyError(
            f"No trainer registered for '{config.foundation_model_name.value}'. "
            f"Available: {[m.value for m in _REGISTRY]}"
        )
    return trainer_cls(config)
