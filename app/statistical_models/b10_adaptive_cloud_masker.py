"""
Adaptive Cloud masking Gaussian Mixture Model for B10
"""

import logging
from typing import Union
import numpy as np
from sklearn.mixture import GaussianMixture

from app.abstract_classes.ml_model import MlModel
from app.models.base_models.base_model import BaseModel
from app.models.intermediate_concepts.adaptive_cloud_masker_response import (
    AdaptiveCloudMaskerResponse,
)

DEFAULT_COMPNENT_COUNT = 5
ADAPTIVE_COMPONENT_COUNT = 3

logger = logging.getLogger("B10AdaptiveCloudMasker")
logger.setLevel(logging.INFO)


class B10AdaptiveCloudMasker(MlModel):
    """
    Trains the B10 Adaptive Cloud masker model.
    """

    def __init__(
        self, base_model: BaseModel = BaseModel.ALLOTROPE_B10_ADAPTIVE_CLOUD_MASKER
    ):
        """
        Class Constructor
        """
        super().__init__(base_model=base_model)
        self.expansive_percentiles = None
        self.restrictive_percentiles = None
        self.significant_cloud_potential_in_celsius: float = None
        self.physical_cloud_threshold_in_celsius: float = None
        self.sampling_ratio: float = None
        self.n_comp: int = None
        self.anchors = None
        self.model = None
        self.sample_count: int = None
        self.probe = None

    def configure(self, sampling_ratio: float = 0.1, **kwargs):
        """
        Configure the model
        """
        self.expansive_percentiles = [2, 8, 50, 92, 98]
        self.restrictive_percentiles = [2, 8, 50]
        self.significant_cloud_potential_in_celsius: float = 0.0
        self.physical_cloud_threshold_in_celsius: float = 30.0
        self.sampling_ratio: float = sampling_ratio

    def train(
        self, input_cube: Union[np.ndarray, np.ma.MaskedArray], **kwargs
    ):  # Pylint
        """
        Trains the model.

        Inputs are in celsius always.
        """
        if isinstance(input_cube, np.ma.MaskedArray):
            print("Masked Array")
            valid_pixels = input_cube.compressed().reshape(-1, 1)
        elif isinstance(input_cube, np.ndarray):
            print("Normal Array")
            valid_pixels = input_cube.reshape(-1, 1)
        else:
            raise TypeError("Unsupported Data Type")

        print(valid_pixels.shape)
        # First we probe the distribution and get the physics of the scene
        probe = np.percentile(valid_pixels, self.expansive_percentiles)
        print(probe)
        self.probe = probe

        # We then apply a high temperature clip for stability
        # This will prevent high temperature anomalies from poisoning the cloud means
        p95 = np.percentile(valid_pixels, 95)
        training_data = valid_pixels[valid_pixels <= p95].reshape(-1, 1)

        # We then perform a conditional set up
        # If the second percentile is freezing then therre is significant cloud potential
        if probe[0] < self.significant_cloud_potential_in_celsius:
            self.n_comp: int = DEFAULT_COMPNENT_COUNT
            self.anchors = probe.reshape(-1, 1)
        else:
            self.n_comp: int = DEFAULT_COMPNENT_COUNT
            # PHYSICS-DRIVEN: Force the GMM to look for cold clusters
            # even if they aren't in the P2/P8 range.
            # This prevents the anchors from 'drifting' too far warm.
            self.anchors = np.array(
                [
                    -10.0,  # Force anchor for Ice Clouds
                    5.0,  # Force anchor for Warm Clouds
                    probe[2],  # P50: Median Land
                    probe[3],  # P92: Hot Land
                    probe[4],  # P98: Anomaly
                ]
            ).reshape(-1, 1)

        # Fit the GMM after sampling

        self.sample_count: int = int(training_data.shape[0] * self.sampling_ratio)
        logger.info("Sample Count set to : %d", self.sample_count)

        # Create a random number generator
        rng = np.random.default_rng()
        sampled_data = rng.choice(training_data, size=self.sample_count, replace=False)

        # Train the model
        self.model = GaussianMixture(
            n_components=self.n_comp, means_init=self.anchors, random_state=42
        )
        self.model.fit(sampled_data)

    def predict(
        self, input_cube: Union[np.ndarray, np.ma.MaskedArray], **kwargs
    ) -> AdaptiveCloudMaskerResponse:
        """
        Perform actual prediction
        """
        if self.model is None:
            raise ValueError("Model has not yet been fit")

        if isinstance(input_cube, np.ma.MaskedArray):
            valid_pixels = input_cube.compressed().reshape(-1, 1)
        elif isinstance(input_cube, np.ndarray):
            valid_pixels = input_cube.reshape(-1, 1)
        else:
            raise TypeError("Unsupported Data Type")
        labels_1d = self.model.predict(valid_pixels)

        # Perform physical verification by masking clusters that are actually cold
        cluster_means = self.model.means_.flatten()
        scene_median = self.probe[2]
        dynamic_threshold = scene_median - 12.0
        print(dynamic_threshold)
        cloud_indices = np.where(cluster_means < dynamic_threshold)[0]
        print(cloud_indices)

        # Create the spatial grid
        label_grid = np.full(input_cube.shape, -1, dtype=np.int8)
        if isinstance(input_cube, np.ma.MaskedArray):
            label_grid[~input_cube.mask] = labels_1d
        elif isinstance(input_cube, np.ndarray):
            # For normal ndarray, mask is False everywhere
            label_grid[:] = labels_1d.reshape(input_cube.shape)

        # Create the final boolean mask
        is_cloud = np.isin(label_grid, cloud_indices)

        return AdaptiveCloudMaskerResponse(
            cloud_mask=is_cloud,
            n_comp=self.n_comp,
            model=self.model,
            anchors=self.anchors,
            pixels_masked=is_cloud.sum(),
        )
