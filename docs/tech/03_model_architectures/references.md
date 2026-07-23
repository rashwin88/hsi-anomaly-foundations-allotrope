# References

Core papers cited across Chapter 3.

## Transformers and attention

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017. Original scaled dot-product
  attention; introduced the `1/sqrt(d_k)` scaling and the FFN expansion ratio convention.
- Dosovitskiy et al., *An Image is Worth 16x16 Words (ViT)*, ICLR 2021. Patch-embedding
  approach that motivates `OverlapPatchEmbedding`.
- Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with
  Transformers*, NeurIPS 2021. Source of the overlap-patch + efficient attention + Mix-FFN
  hierarchy used in Chakshu and Indradhanu.
- Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020.
  Pre-Norm vs. Post-Norm analysis; justifies the residual-stream pattern in
  `SegFormerBlock`.

## Masked autoencoders

- He et al., *Masked Autoencoders Are Scalable Vision Learners (MAE)*, CVPR 2022. The
  token-removal recipe used in `SegFormerEncoder` Stage 1 and `TokenMasking`.

## Normalization and activations

- Ioffe & Szegedy, *Batch Normalization*, ICML 2015. Used in `SpatialEncoderBlock`,
  `SpatialDecoderBlock`, and `SpectralCompressor`.
- Hendrycks & Gimpel, *Gaussian Error Linear Units (GELU)*, 2016. Smooth nonlinearity used
  throughout the transformer and conv blocks.

## Upsampling and decoders

- Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient
  Sub-Pixel Convolutional Neural Network (PixelShuffle)*, CVPR 2016. The final upsampling
  trick in `SegFormerDecoder`.
- Radford et al., *Unsupervised Representation Learning with Deep Convolutional Generative
  Adversarial Networks (DCGAN)*, ICLR 2016. Source of the "no BN/activation on the final
  conv" trick used in `SpatialDecoder` and `SpectralDecompressor`.
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks
  (pix2pix)*, CVPR 2017. Reinforces the same final-layer convention.
- Odena et al., *Deconvolution and Checkerboard Artifacts*, distill.pub 2016. Background on
  why `K = 4, S = 2` (kernel divisible by stride) avoids the classic ConvTranspose
  checkerboard.
- Aitken et al., *Checkerboard artifact free sub-pixel convolution: A note on sub-pixel
  convolution, resize convolution and convolution resize*, 2017. ICNR initialization for
  PixelShuffle, useful background even though Allotrope does not use it explicitly.

## Autoencoders and anomaly detection

- Hinton & Salakhutdinov, *Reducing the Dimensionality of Data with Neural Networks*,
  Science 2006. The original autoencoder paper; intellectual foundation for Pratibimba.
- Bergmann et al., *Improving Unsupervised Defect Segmentation by Applying Structural
  Similarity*, 2018. Modern reformulation of autoencoder-based anomaly detection.
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*,
  MICCAI 2015. Encoder-decoder structure; the Allotrope spatial AE is a U-Net without skip
  connections.

## Imputation networks

- Yoon et al., *GAIN: Missing Data Imputation using Generative Adversarial Nets*, ICML
  2018. Background on the indicator-channel trick used by Asanskrita and Drashta.

## Hyperspectral classical methods

- Kruse et al., *The Spectral Image Processing System (SIPS) — Interactive Visualization
  and Analysis of Imaging Spectrometer Data*, Remote Sensing of Environment, 1993. Origin
  of Spectral Angle Mapper (SAM), implemented in `SAMLoss`.
- Green et al., *A Transformation for Ordering Multispectral Data in Terms of Image
  Quality with Implications for Noise Removal*, IEEE TGRS, 1988. Origin of MNF; the
  intellectual cousin of the learned `SpectralCompressor`.

## Codename source

- *project_model_codenames.md* in the project memory directory documents the seven
  Sanskrit codenames (Pratibimba, Antardhana, Tirohita, Asanskrita, Drashta, Chakshu,
  Indradhanu) and the slug they map to.
