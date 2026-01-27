### Image Transformations

The general idea is to have a set of transformations that can be applied to an image in a specific dimensional arrangement to any other dimensional arrangement. A specific utility class `ImageCubeOperations` is provided to handle this.

The class structure and relationships are as follows:

<p align="center">
  <img src="assets/image_transforms.png" width="700" alt="File Handling Flow">
</p>

The class can take in *both numpy arrays and tensors* and will return *both numpy arrays and tensors*. Internally it will use the `torch` library to perform the transformations.