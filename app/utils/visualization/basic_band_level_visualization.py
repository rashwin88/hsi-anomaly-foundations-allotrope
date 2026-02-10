from matplotlib import pyplot as plt
from typing import List, Dict
import math
import numpy as np


from app.models.images.cube_representation import CubeRepresentation
from app.models.file_processing.sources import FileSourceConfig
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.utils.files.he5_helper import HE5Helper
from app.utils.files.tif_helper import TIFHelper
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier
from app.models.hyperspectral_concepts.spectral_family import SpectralFamily

DEFAULT_VIS_PATH = "sample_visualization"


class BasicBandLevelVisualizationHE5:
    """
    Constructs a basic band level visualization of a cube.
    An input band number will have to be specified.
    """

    def __init__(self, file_source_config: FileSourceConfig):
        """

        Accepts a file source config and constructs a basic band level visualization.
        """
        # initialize the he5 helper
        self.helper = HE5Helper(
            file_source_config=file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.PRISMA_HYPERSPECTRAL),
        )
        self.image_cube_operations = ImageCubeOperations()
        self.max_cols = 3

    def visualize_band(
        self, band_numbers: List[int], spectral_family: SpectralFamily, file_name: str
    ):
        """
        Visualizes a band of a cube.
        """
        # Get the number of plots
        num_plots = len(band_numbers)
        num_rows = math.ceil(num_plots / self.max_cols)

        # Set up the plot
        fig, axes = plt.subplots(
            num_rows,
            self.max_cols,
            figsize=(4 * self.max_cols, 4.5 * num_rows),
            constrained_layout=True,
            dpi=120,
        )
        fig.suptitle(
            f"Hyperspectral band analysis \n {spectral_family.value}",
            fontsize=14,
            fontweight="bold",
        )
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
        plot_images = []

        for i, ax in enumerate(axes_flat):
            if i < num_plots:
                single_band = self.helper.extract_specific_bands(
                    bands=[band_numbers[i]],
                    masking_needed=True,
                    spectral_family=spectral_family,
                    mode="specific",
                )
                # Transform the image to BIP
                single_band = self.image_cube_operations.convert_cube(
                    single_band,
                    from_format=CubeRepresentation.BIL,
                    to_format=CubeRepresentation.BIP,
                )
                print(type(single_band))
                im = ax.imshow(single_band, cmap="Spectral")
                plot_images.append(im)
                label_text = f"Band : {band_numbers[i]}"
                ax.set_title(label_text, fontsize=11, fontfamily="serif", pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                    spine.set_linewidth(0.5)
            else:
                ax.axis("off")
        # visualize the band
        if plot_images:
            cbar = fig.colorbar(
                plot_images[-1],
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.02,
            )
            cbar.set_label("Reflectance Intensity", fontsize=10)
        plt.savefig(f"{DEFAULT_VIS_PATH}/{file_name}_multibands.jpg")


class BasicBandLevelVisualizationTIF:
    """
    Constructs a basic band level visualization of a cube.
    An input band number will have to be specified.
    """

    def __init__(self, file_source_config: FileSourceConfig):
        """

        Accepts a file source config and constructs a basic band level visualization.
        """
        # initialize the he5 helper
        self.helper = TIFHelper(
            file_source_config=file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.LANDSAT_THERMAL),
        )
        self.image_cube_operations = ImageCubeOperations()
        self.max_cols = 3

    def visualize_band(self, band_numbers: List[int], file_name: str):
        """
        Visualizes a band of a cube.
        """
        # Get the number of plots
        num_plots = len(band_numbers)
        num_rows = math.ceil(num_plots / self.max_cols)

        # Set up the plot
        fig, axes = plt.subplots(
            num_rows,
            self.max_cols,
            figsize=(4 * self.max_cols, 4.5 * num_rows),
            constrained_layout=True,
            dpi=120,
        )
        fig.suptitle(
            f"Thermal band analysis",
            fontsize=14,
            fontweight="bold",
        )
        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        plot_images = []

        # get the band from the file
        # A short note here - we are pulling out the bands directly from the file.
        cube = self.helper.extract_specific_bands(
            bands=band_numbers, mode="specific", masking_needed=True
        )

        # However, we have to be careful here because the bands in TIF start at 1 and since we
        # are pulling them out specifically, the band indexes will get reset and we will have to map them back.
        # This is why we use a band_mapping dictionary
        band_mapping: Dict[int, int] = {i: band for i, band in enumerate(band_numbers)}

        # We have now extracted the bands,
        # Cast the cube to BIP format
        # By default TIF files are in BSQ format.
        # We convert them to BIP Format for visualization
        cube = self.image_cube_operations.convert_cube(
            cube, CubeRepresentation.BSQ, CubeRepresentation.BIP, output_form="numpy"
        )

        for i, ax in enumerate(axes_flat):
            if i < num_plots:
                single_band = cube[:, :, i]
                im = ax.imshow(single_band, cmap="plasma")
                plot_images.append(im)
                label_text = f"Band : {band_mapping[i]}"
                ax.set_title(label_text, fontsize=11, fontfamily="serif", pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                    spine.set_linewidth(0.5)
            else:
                ax.axis("off")

        # visualize the band
        if plot_images:
            cbar = fig.colorbar(
                plot_images[-1],
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.02,
            )
            cbar.set_label("Thermal Temperature", fontsize=10)
        plt.savefig(f"{DEFAULT_VIS_PATH}/{file_name}_multibands.jpg")


if __name__ == "__main__":
    file_source_config = FileSourceConfig(
        source_path="tests/test_payloads/thermal_1/LC09_L2SP_150044_20251009_20251010_02_T1_ST_B10.TIF"
    )
    basic_band_level_visualization = BasicBandLevelVisualizationTIF(file_source_config)
    basic_band_level_visualization.visualize_band(
        [1],
        "new_thermal_single_band_1",
    )
    file_source_config = FileSourceConfig(
        source_path="tests/test_payloads/phase_2/Set-1/Hypersepctral Datasets/PRS_L2D_STD_20201214060713_20201214060717_0001.he5"
    )
    basic_band_level_visualization = BasicBandLevelVisualizationHE5(file_source_config)
    basic_band_level_visualization.visualize_band(
        [14, 19, 24, 29, 34, 39, 44, 49, 54, 59],
        SpectralFamily.VNIR,
        "new_vnir_multibands_14_59",
    )
