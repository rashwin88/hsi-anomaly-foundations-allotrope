"""
Band mosaics for eyeballing an EnMAP scene during development.

The EnMAP counterpart to basic_band_level_visualization.py - see that module for
why these exist and why they are NOT the product's rendering path.

EnMAP differs in that a scene is a folder, not a single file: the cube lives in
*-SPECTRAL_IMAGE.TIF alongside quality-mask rasters and METADATA.XML. This reads
the cube via EnmapHelper and pulls wavelengths from EnmapXmlParser, so band
labels show real nanometres rather than bare indices.

Requires matplotlib and writes to sample_visualization/. Keep it out of any
import path the api touches.
"""

import math
import random
from typing import List

import numpy as np
from matplotlib import pyplot as plt

from app.models.file_processing.sources import FileSourceConfig
from app.utils.files.enmap_helper import EnmapHelper
from app.utils.files.enmap_xml_parser import EnmapXmlParser
from app.templates.template_mappings import TEMPLATE_MAPPINGS, TemplateIdentifier

DEFAULT_VIS_PATH = "sample_visualization"


class BasicBandLevelVisualizationEnMAP:
    """
    Constructs a basic band level visualization for EnMAP L2A scenes.
    Reads directly from the folder-based scene and displays individual bands.
    """

    def __init__(self, file_source_config: FileSourceConfig):
        self.helper = EnmapHelper(
            file_source_config=file_source_config,
            template=TEMPLATE_MAPPINGS.get(TemplateIdentifier.ENMAP_HYPERSPECTRAL),
        )
        self.max_cols = 5

    def visualize_band(self, band_numbers: List[int], file_name: str):
        """
        Visualizes specified bands (1-based) from the EnMAP spectral image.
        Band titles include center wavelength from metadata.
        """
        num_plots = len(band_numbers)
        num_rows = math.ceil(num_plots / self.max_cols)

        fig, axes = plt.subplots(
            num_rows,
            self.max_cols,
            figsize=(4 * self.max_cols, 4.5 * num_rows),
            constrained_layout=True,
            dpi=120,
        )
        fig.suptitle(
            "EnMAP L2A Hyperspectral Band Analysis",
            fontsize=14,
            fontweight="bold",
        )

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        # Read all requested bands at once (BSQ: num_bands x H x W)
        cube = self.helper.extract_specific_bands(
            bands=band_numbers, mode="specific", masking_needed=True
        )

        # Get wavelength info for labels
        band_chars = self.helper.file_metadata.band_characterisation
        nodata_value = self.helper.masked_pixel_value

        plot_images = []
        for i, ax in enumerate(axes_flat):
            if i < num_plots:
                band_idx = band_numbers[i]  # 1-based
                single_band = cube[i]  # (H, W)

                # Mask nodata for display
                display_band = np.ma.masked_where(
                    single_band == nodata_value, single_band
                )

                # Apply gain to show reflectance-scaled values
                display_band = display_band * 0.0001

                im = ax.imshow(display_band, cmap="Spectral")
                plot_images.append(im)

                wavelength = band_chars[band_idx - 1].wavelength_center
                label_text = f"Band {band_idx}\n{wavelength:.1f} nm"
                ax.set_title(label_text, fontsize=10, fontfamily="serif", pad=8)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("#333333")
                    spine.set_linewidth(0.5)
            else:
                ax.axis("off")

        if plot_images:
            cbar = fig.colorbar(
                plot_images[-1],
                ax=axes,
                orientation="horizontal",
                fraction=0.05,
                pad=0.02,
            )
            cbar.set_label("Surface Reflectance", fontsize=10)

        output_path = f"{DEFAULT_VIS_PATH}/{file_name}.jpg"
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    random.seed(42)
    band_numbers = sorted(random.sample(range(1, 225), 10))
    print(f"Randomly selected bands: {band_numbers}")

    file_source_config = FileSourceConfig(
        source_path="tests/test_payloads/ENMAP01-____L2A-DT0000059367_20240128T063655Z_018_V010506_20260305T173243Z"
    )
    viz = BasicBandLevelVisualizationEnMAP(file_source_config)
    viz.visualize_band(band_numbers, "enmap_random_10_bands")
