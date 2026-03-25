### Foundation Model Inference System

#### Motivation

After training a foundation model, we need a consistent way to run inference across different models. The inference system mirrors the training system's design: config-driven, model-agnostic, and built on an abstract base class that concrete inferencers extend.

For autoencoder-based anomaly detection, we need reconstructions where each pixel was predicted from its neighbours, never from itself. The checkerboard masking strategy achieves this.

#### Inference Configuration

`InferenceConfig` specifies everything needed to run inference:

```
InferenceConfig
├── foundation_model_name: FoundationModelName (enum)
├── model_config: ModelSpecificConfig (discriminated union)
├── checkpoint_path: str (path to .pt checkpoint)
├── patch_size: int
├── stride: int | None (defaults to patch_size // 2)
├── checkerboard_cell_size: int (default 1)
└── device: str | None (None = auto-detect)
```

- `stride` controls overlap for full-scene inference. Smaller stride = more overlap = smoother results but slower.
- `checkerboard_cell_size` controls the granularity of the checkerboard mask. 1 = single pixel alternation, 2 = 2x2 blocks, etc.

#### Abstract Base: FoundationInferencer

The abstract class handles device setup, model instantiation, checkpoint loading, and sets the model to eval mode. Concrete inferencers implement two methods:

- `build_model()` — instantiate the nn.Module from config
- `infer(tensor, mask)` — model-specific inference logic

The base class provides:

- `predict(tensor, mask)` — moves inputs to device, calls `infer()` under `torch.no_grad()`
- `_load_weights(path)` — loads `model_state_dict` from a training checkpoint

#### Init Flow

```
__init__(config)
  ├── Set device (explicit or auto-detect)
  ├── build_model() → abstract, implemented per model
  ├── _load_weights(checkpoint_path)
  └── model.eval()
```

#### Checkerboard Masking (SpatialAutoencoderInferencer)

The spatial autoencoder reconstructs patches from masked input. To get an honest reconstruction for every pixel (one that was never conditioned on the pixel itself), we use a two-pass checkerboard strategy:

**Pass 1:** Null all cells where checkerboard = 1. The model reconstructs those cells from the remaining context.

**Pass 2:** Null all cells where checkerboard = 0 (the inverse). The model reconstructs those cells.

**Combine:** Each pixel's final reconstruction comes from the pass where it was hidden.

```
Checkerboard (cell_size=2, 8x8):

  0 0 1 1 0 0 1 1      Pass 1: null the 1s     Pass 2: null the 0s
  0 0 1 1 0 0 1 1      model reconstructs 1s   model reconstructs 0s
  1 1 0 0 1 1 0 0
  1 1 0 0 1 1 0 0      Final: pixel's value = reconstruction from
  0 0 1 1 0 0 1 1      the pass where it was hidden
  0 0 1 1 0 0 1 1
  1 1 0 0 1 1 0 0
  1 1 0 0 1 1 0 0
```

The checkerboard is built via broadcasting:

```python
rows = arange(H) // cell_size   # e.g. [0,0,1,1,2,2,3,3]
cols = arange(W) // cell_size
grid = (rows[:, None] + cols[None, :]) % 2
```

`rows[:, None]` is shape (H,1), `cols[None,:]` is shape (1,W). Addition broadcasts to (H,W). The `% 2` creates alternating parity — same-parity blocks get 0, different-parity blocks get 1.

#### Full-Scene Inference

`predict_full_scene(scene, mask)` handles scenes larger than the patch size:

1. Build a `PatchingPlan` via `PatchPlanGenerator` — computes sliding window coordinates with edge snapping
2. For each patch coordinate, extract the patch and its mask
3. Run `predict()` (checkerboard reconstruction) on each patch
4. Accumulate reconstructions and valid-pixel counts in full-frame buffers
5. Overlap-average: divide accumulated reconstructions by the count at each pixel

The stride controls how much patches overlap. With `stride = patch_size // 2`, each interior pixel is covered by 4 patches, giving a smooth averaged reconstruction.

#### Downstream Usage

The inferencer returns pure reconstructions. The caller computes anomaly scores:

```python
inferencer = get_inferencer(config)
reconstruction = inferencer.predict_full_scene(scene, mask)
anomaly_map = (reconstruction - scene) ** 2  # per-pixel MSE
```

#### Adding a New Inferencer

1. Create a config class in `training_config.py` (if not already present)
2. Create a concrete inferencer in `app/foundation_models/inferencers/`
3. Implement `build_model()` and `infer(tensor, mask)`
4. Register it in `inferencer_factory.py`

#### File Map

```
app/
├── abstract_classes/
│   └── foundation_inferencer.py        # Abstract base class
├── models/training/
│   └── inference_config.py             # InferenceConfig Pydantic model
└── foundation_models/inferencers/
    ├── inferencer_factory.py           # Registry + get_inferencer()
    └── spatial_autoencoder_inferencer.py  # Checkerboard reconstruction
```
