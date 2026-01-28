from app.models.images.cube_representation import CubeRepresentation
from app.models.file_processing.file_metadata_models import FullMetadata
from app.models.file_processing.sources import FileSourceConfig
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations
from app.utils.files.he5_helper import HE5Helper
from matplotlib import pyplot as plt
from typing import List
import math
import numpy as np

DEFAULT_VIS_PATH = "sample_visualization"


class BasicBandLevelVisualization:
    """
    Constructs a basic band level visualization of a cube.
    An input band number will have to be specified.
    """

    def __init__(self, file_source_config: FileSourceConfig):
        """

        Accepts a file source config and constructs a basic band level visualization.
        """
        # initialize the he5 helper
        self.helper = HE5Helper(file_source_config)
        self.image_cube_operations = ImageCubeOperations()
        self.max_cols = 3

    def visualize_band(
        self, band_numbers: List[int], file_reference: str, file_name: str
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
            f"Hyperspectral band analysis \n {file_reference}",
            fontsize=14,
            fontweight="bold",
        )
        axes_flat = axes.flatten() if num_plots > 1 else [axes]
        plot_images = []

        # get the band from the file
        cube = self.helper.get_dataset(file_reference)
        # Cast the cube to BIP format
        cube = self.image_cube_operations.convert_cube(
            cube, CubeRepresentation.BIL, CubeRepresentation.BIP, output_form="numpy"
        )

        for i, ax in enumerate(axes_flat):
            if i < num_plots:
                single_band = cube[:, :, band_numbers[i]]
                masked_band = np.ma.masked_where(single_band == 0, single_band)
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
        plt.show()


if __name__ == "__main__":
    file_source_config = FileSourceConfig(
        source_path="raw_files/Hyper/PRS_L2D_STD_20231229050902_20231229050907_0001.he5"
    )
    basic_band_level_visualization = BasicBandLevelVisualization(file_source_config)
    basic_band_level_visualization.visualize_band(
        [11, 21, 31, 41, 51, 61, 62],
        "HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube",
        "sample_swir_multibands_1_10",
    )
