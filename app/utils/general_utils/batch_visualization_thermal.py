"""
Used to visualize a thermal batch of patches.
there are no band wavelengths as a concept in thermal so visualizations
will proceed off of B10, the first and only band
"""

# We have a webdataset created and the objective is now to 

import torch
import webdataset as wds
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

# Import the shard pipe expression builder for pulling in shards
from app.utils.general_utils.shard_pipe_expression_builder import shard_pipe_expression_builder
from app.utils.image_transformation.image_cube_operations import ImageCubeOperations, CubeRepresentation

S3_BUCKET : str = "allotrope-raw-data-india"

def visualize_thermal_patches(
        patch_location_key: str,
        batch_size: int = 16,
        row_count:int = 4,
        file_name: str = None,
        bucket: str = S3_BUCKET
):
    """
    Visualizes thermal patches and writes as a png file to a specific location.
    """
    # Get the converter ready
    converter = ImageCubeOperations()

    # Construct the shard url
    shard_url = shard_pipe_expression_builder(
        data_key=patch_location_key,
        bucket_name=bucket
    )

    print(shard_url)
    # Build out the dataset
    dataset = (
        wds.WebDataset(shard_url, shardshuffle=10).
        decode()
    )

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2
    )

    # Get the batch
    batch = next(iter(dataloader))

    # From the batch, pull out the pixels and the validity cube
    pixels = batch.get("pixels.npy")
    validity_cube = batch.get("validity_cube.npy")

    # Convert the slices to a single grid
    pixel_grid = make_grid(pixels, nrow=row_count, padding=4)
    validity_grid = make_grid(validity_cube, nrow=row_count, padding=4)

    # Convert to BIP
    pixel_bip = converter.convert_cube(pixel_grid, CubeRepresentation.BSQ, CubeRepresentation.BIP)
    validity_bip = converter.convert_cube(validity_grid, CubeRepresentation.BSQ, CubeRepresentation.BIP)

    # Take only a single channel as make_grid defaults to 3 channels
    pixel_2d = pixel_bip[: , :, 0]
    validity_2d = validity_bip[: , :, 0]

    # Construct the masked grid
    masked_grid = np.ma.masked_where(
        validity_2d==0,
        pixel_2d
    )

    # Build the display
    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(masked_grid, cmap="viridis")
    ax.axis("off")
    ax.set_title("Thermal Patches (B10) — Masked")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized Temperature")
    plt.tight_layout()

    if file_name:
        plt.savefig(file_name, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return batch

if __name__ == "__main__":
    base_batch = visualize_thermal_patches(
        patch_location_key="patches/final/landsat/512-pixel-training/",
        batch_size=32,
        row_count=8,
        file_name="thermal_64_exp1.png"
    )
    