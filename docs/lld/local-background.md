# Local-background scoring

**File:** `app/detectors/_local_background.py`
**Used by:** `LocalRXDetector`, `MNFCompressionLRXDetector`

## Purpose

Local RX scores each pixel against a ring of its own neighbours rather than against the
whole scene. That needs one covariance matrix, one inverse and one quadratic form **per
pixel**. This module makes that affordable.

## The problem it solves

Global RX computes one covariance for the entire image. Local RX cannot: the whole point is
that the background changes across the scene, so a warm spot inside an already-warm field
is still detectable.

The cost is brutal. A PRISMA scene is roughly 1210 by 1219, about 1.47 million pixels. Doing
this one pixel at a time in numpy means 1.47 million calls to `np.linalg.solve`, each on a
matrix small enough that Python's per-call overhead dominates the actual arithmetic.

Batching turns that into a handful of calls on `(N, B, B)` tensors, which a GPU can
saturate and which even CPU BLAS handles far better than a Python loop.

## Interfaces

```python
def select_device() -> torch.device          # re-export of get_device
def batch_mahalanobis(
    X_bg_padded: np.ndarray,   # (N, max_bg, B) float64
    n_bg_arr:    np.ndarray,   # (N,) int64  - real background count per pixel
    x_test:      np.ndarray,   # (N, B) float64
    count:       int,          # how many rows of the batch are valid
    B:           int,
    reg:         float,
    device:      torch.device,
) -> np.ndarray                # (count,) float64 scores
```

## Why the padding argument exists

Background counts vary per pixel. A pixel in open ground has a full annulus; a pixel near
the swath edge, or beside a cloud-masked region, has fewer valid neighbours. Tensors need
rectangular shapes, so every row is padded to the widest case and `n_bg_arr` says how much
of each row is real.

Everything downstream is masked accordingly. Work through a two-pixel batch with three
bands, where pixel 0 has 4 valid neighbours and pixel 1 has only 2, padded to `max_bg = 4`:

```
mask = arange(4) < n[:, None]

pixel 0 (n=4):  [0,1,2,3] < 4  ->  [True,  True,  True,  True ]
pixel 1 (n=2):  [0,1,2,3] < 2  ->  [True,  True,  False, False]
```

The mean divides by the real count, not by `max_bg`:

```
counts_f = mask.sum(dim=1)   ->  pixel 0: 4     pixel 1: 2
mu       = (X * mask).sum(dim=1) / counts_f
```

Had it divided by 4, pixel 1's mean would be understated by half, because two of the four
slots contribute zero. The centred data is masked again for the same reason, so padding
contributes nothing to the covariance either.

## The ridge term

```python
cov += reg * torch.eye(B, device=device, dtype=dtype)
```

A local background is small. With `outer_window=25` and `inner_window=5`, the annulus holds
at most:

```
(2 * 25 + 1)^2  =  51 * 51  =  2601   outer square
(2 *  5 + 1)^2  =  11 * 11  =   121   inner guard, excluded
                              ------
                               2480   background pixels
```

2480 samples for a covariance that is `B x B`. On the 165-band common grid that is fine; on
a compressed MNF cube of 10 components it is comfortable. But near an edge the real count
can fall to a few dozen, and a covariance estimated from fewer samples than it has
dimensions is singular. `reg` (default `1e-4`) keeps the solve well posed.

**Note the default disagrees with one of the old docs.** `DEFAULT_REGULARIZATION = 1e-4` in
`local_rx_detector.py`; a since-deleted doc claimed `1e-3`. The code is authoritative.

## Precision

```python
dtype = torch.float32 if device.type != "cpu" else torch.float64
```

float32 on GPU for speed, float64 on CPU for precision. This means **the same scene can
produce slightly different scores on different hardware**. The difference is far below any
threshold anyone sets, but it does mean scores are not bit-comparable across machines, and
a test asserting exact equality would fail when it moved from a laptop to the GPU box.

## Invariants

- `X_bg_padded[i, :n_bg_arr[i]]` are real measurements; everything past `n_bg_arr[i]` is
  padding and must never influence a result.
- `count <= X_bg_padded.shape[0]`. The caller sizes the batch; this function trusts it.
- The return is always float64 on CPU, whatever the compute dtype was.

## Decisions

**Why `solve` rather than `inv`.** `torch.linalg.solve(cov, x)` computes the same thing as
`inv(cov) @ x` but is better conditioned and faster. Forming an explicit inverse is the
thing you were taught not to do in numerical linear algebra, and this is why.

**Why device selection is not defined here.** It was, briefly, and that was a mistake:
`app/utils/torch_helpers/device_selection.get_device` already existed and was used by four
other modules. Defining a second one here made three implementations across the repo. The
module now re-exports the canonical one so the detectors still have a single import.

## Verification

This function has a genuine numerical check rather than an import test. The batched result
was compared against a plain numpy reference implementation over five pixels with varying
background counts:

```
max absolute difference: 3.55e-15
```

That is floating-point noise, and it confirms the masking arithmetic above is right - a
mean divided by `max_bg` instead of the real count would show up immediately as a large
difference on the short rows.
