# 26 · Counting all 5.5 million parameters

> **The one thing this part teaches:** you can derive the exact size of this
> model, layer by layer, with three formulas and some patience — and it comes out
> right to the last digit.

This part is an exercise as much as a reference. Every number below is worked by
hand and then checked against the `torchinfo` table at the bottom of
[`research/model_break_down/05_hyperspectral_segformer_mae.md`](../research/model_break_down/05_hyperspectral_segformer_mae.md).

They all match. If you can reproduce this table, you understand the architecture.

---

## First: what is a parameter?

A parameter is a single number the training process adjusts. A weight or a bias.

When somebody says "a 5.5 million parameter model", they mean there are 5.5
million individual numbers stored in the file, each one nudged thousands of
times during training.

For context: the file is about 22 MB, because each parameter is a 32-bit float
(4 bytes). `5,506,629 x 4 = 22,026,516 bytes`. That matches the `torchinfo`
report of "Params size (MB): 22.03".

---

## The three formulas

**A convolution.** One weight per (input channel, output channel, kernel row,
kernel column), plus one bias per output channel:

```
Conv2d(cin, cout, k)  :  cin * cout * k * k  +  cout
```

**A depthwise convolution.** Each channel has its own kernel and there is no
cross-channel mixing, so drop one factor:

```
Conv2d depthwise (groups=c)  :  c * k * k  +  c
```

**A linear layer or a LayerNorm.**

```
Linear(a, b)   :  a * b  +  b
LayerNorm(c)   :  2 * c            (one scale and one shift per feature)
```

Note a 1x1 convolution and a linear layer are the same formula, since `k*k = 1`.

---

## The configuration we are counting

```
C = 165                      input bands
D = 32                       compressed bands
embed_dims  = [32, 64, 160, 256]
num_heads   = [2, 2, 5, 8]           (does not affect the count)
R           = [8, 4, 2, 1]
num_blocks  = [2, 2, 2, 2]
decoder_dim = 256
E = 4                        Mix-FFN expansion
```

---

## Section 1 · The front end

```
SpectralCompressor
    Conv2d(165, 32, k=1)   : 165 * 32 * 1 * 1 + 32
                           = 5,280 + 32
                           = 5,312
    BatchNorm2d(32)        : 2 * 32
                           = 64
                                          ------
                              subtotal      5,376
```

---

## Section 2 · The four patch embeddings

Each is a convolution plus a LayerNorm.

**Stage 1** — `Conv2d(32, 32, k=4)`:

```
32 * 32 * 4 * 4 = 32 * 32 * 16 = 16,384
             + bias 32          =     32
                                  -------
                                   16,416
LayerNorm(32) = 2 * 32          =      64
                                  -------
                                   16,480
```

**Stage 2** — `Conv2d(32, 64, k=3)`:

```
32 * 64 * 9 = 18,432   + 64  = 18,496
LayerNorm(64) = 128
                               18,624
```

**Stage 3** — `Conv2d(64, 160, k=3)`:

```
64 * 160 * 9 = 92,160  + 160 = 92,320
LayerNorm(160) = 320
                               92,640
```

**Stage 4** — `Conv2d(160, 256, k=3)`:

```
160 * 256 * 9 = 368,640 + 256 = 368,896
LayerNorm(256) = 512
                                369,408
```

```
subtotal: 16,480 + 18,624 + 92,640 + 369,408 = 497,152
```

---

## Section 3 · The transformer blocks

One block = attention (ESA) + Mix-FFN + two LayerNorms.

- **ESA** = four `Linear(C, C)` (q, k, v, output projection), plus — when
  `R > 1` — a `Conv2d(C, C, k=R, s=R)` and a `LayerNorm(C)`.
- **Mix-FFN** = `Linear(C, 4C)` + depthwise `Conv2d(4C, 4C, k=3)` +
  `Linear(4C, C)`.

### Stage 1 (`C = 32`, `R = 8`)

```
attention
    q, k, v, proj   4 * (32*32 + 32)   = 4 * 1,056  =  4,224
    reduction conv  32*32*8*8 + 32     = 65,536+32  = 65,568
    reduction norm  2 * 32                          =     64
                                                      ------
                                            ESA       69,856

Mix-FFN   (hidden = 32 * 4 = 128)
    fc1     32*128 + 128                            =  4,224
    dwconv  128*9 + 128                             =  1,280
    fc2     128*32 + 32                             =  4,128
                                                      ------
                                        Mix-FFN       9,632

two LayerNorms   2 * (2 * 32)                       =    128
                                                      ------
                          one block                   79,616
                          x 2 blocks                 159,232
```

`torchinfo` reports **159,232**. Match.

### Stage 2 (`C = 64`, `R = 4`)

```
attention
    q,k,v,proj      4 * (64*64 + 64)   = 4 * 4,160  = 16,640
    reduction conv  64*64*4*4 + 64     = 65,536+64  = 65,600
    reduction norm  2 * 64                          =    128
                                            ESA       82,368
Mix-FFN   (hidden = 256)
    fc1     64*256 + 256                            = 16,640
    dwconv  256*9 + 256                             =  2,560
    fc2     256*64 + 64                             = 16,448
                                        Mix-FFN      35,648
two LayerNorms   2 * (2 * 64)                       =    256
                          one block                  118,272
                          x 2 blocks                 236,544
```

`torchinfo` reports **236,544**. Match.

### Stage 3 (`C = 160`, `R = 2`)

```
attention
    q,k,v,proj      4 * (160*160 + 160) = 4 * 25,760 = 103,040
    reduction conv  160*160*2*2 + 160   = 102,400+160= 102,560
    reduction norm  2 * 160                          =     320
                                            ESA        205,920
Mix-FFN   (hidden = 640)
    fc1     160*640 + 640                            = 103,040
    dwconv  640*9 + 640                              =   6,400
    fc2     640*160 + 160                            = 102,560
                                        Mix-FFN       212,000
two LayerNorms   2 * (2 * 160)                       =     640
                          one block                   418,560
                          x 2 blocks                  837,120
```

`torchinfo` reports **837,120**. Match.

### Stage 4 (`C = 256`, `R = 1` — no reduction convolution)

```
attention
    q,k,v,proj      4 * (256*256 + 256) = 4 * 65,792 = 263,168
                                            ESA        263,168
Mix-FFN   (hidden = 1024)
    fc1     256*1024 + 1024                          = 263,168
    dwconv  1024*9 + 1024                            =  10,240
    fc2     1024*256 + 256                           = 262,400
                                        Mix-FFN       535,808
two LayerNorms   2 * (2 * 256)                       =   1,024
                          one block                   800,000
                          x 2 blocks                1,600,000
```

`torchinfo` reports **1,600,000**. Match. (A suspiciously round number, and
genuinely a coincidence.)

```
blocks subtotal: 159,232 + 236,544 + 837,120 + 1,600,000 = 2,832,896
```

---

## Section 4 · Stage output norms

One LayerNorm at the end of each stage:

```
2 * (32 + 64 + 160 + 256) = 2 * 512 = 1,024
```

---

## Section 5 · The decoder

```
projections
    32  -> 256, k=1  :   32*256 + 256  =     8,448
    64  -> 256, k=1  :   64*256 + 256  =    16,640
    160 -> 256, k=1  :  160*256 + 256  =    41,216
    256 -> 256, k=1  :  256*256 + 256  =    65,792

fuse
    1024 -> 256, k=1 : 1024*256 + 256  =   262,400

refine
    256 -> 256, k=3  : 256*256*9 + 256 =   590,080
    256 -> 512, k=3  : 256*512*9 + 512 = 1,180,160
                                          ---------
                             subtotal     2,164,736
```

---

## Section 6 · The back end

```
SpectralDecompressor
    Conv2d(32, 165, k=1) : 32 * 165 + 165 = 5,280 + 165 = 5,445
```

---

## The grand total

| Component | Parameters | Share |
|---|---:|---:|
| SpectralCompressor | 5,376 | 0.1% |
| Patch embeddings | 497,152 | 9.0% |
| Transformer blocks | 2,832,896 | 51.4% |
| Stage output norms | 1,024 | 0.0% |
| Decoder | 2,164,736 | 39.3% |
| SpectralDecompressor | 5,445 | 0.1% |
| **Total** | **5,506,629** | 100% |

Adding it up carefully:

```
        5,376
      497,152
    2,832,896
        1,024
    2,164,736
        5,445
    ---------
    5,506,629
```

`torchinfo` reports `Total params: 5,506,629`.

**Exact match, to the last digit.**

---

## But the manifest says 5,507,354 — where do the extra 725 come from?

```json
"params": 5507354
```

```
5,507,354 - 5,506,629 = 725
```

The difference is **buffers** — tensors saved inside the model file that are not
trainable parameters (part 09).

```
PixelNormalize.mean        165
PixelNormalize.std         165
PixelDenormalize.mean      165
PixelDenormalize.std       165
                           ---
                           660

BatchNorm2d running_mean         32
BatchNorm2d running_var          32
BatchNorm2d num_batches_tracked   1
                                 --
                                 65

660 + 65 = 725
```

```
5,506,629 + 725 = 5,507,354
```

Precisely the manifest's number.

Nothing here is guesswork — every term is derivable. It also demonstrates part
09's point concretely: those 660 normalisation numbers genuinely live inside the
checkpoint file, which is why you cannot swap them without rebuilding the model.

> **Where do the BatchNorm buffers come from?** BatchNorm keeps a running average
> of the mean and variance it has seen, so that at evaluation time it can use
> stable statistics instead of the current batch's. `num_batches_tracked` is a
> single counter of how many batches contributed. Three buffers, 65 numbers, for
> the one BatchNorm in the compressor.

---

## Five things this table teaches

**1. Stage 4 is the most expensive stage.** 1.6 million parameters, 29% of the
model — while handling only **16 tokens**. Parameters follow channel *width*,
not token count.

**2. The decoder is 39% of the model.** "Lightweight decoder" is relative to a
segmentation decoder that would run at full resolution. It is not small.

**3. Two layers are 32% of everything.** The decoder's two 3x3 refine
convolutions are 1,770,240 parameters between them.

**4. The spectral bottleneck costs 0.2% and saves far more.** Without it, stage
1's patch embedding alone would be `165*32*16 + 32 = 84,512` rather than 16,416
— and every activation tensor in the encoder would be five times larger.

**5. Parameters are not compute.** `torchinfo` reports **2.32 gigabytes** of
multiply-add operations for a single 128x128 patch. Most of that happens at
stage 1, where the parameters are fewest but the tensors are largest. A layer
with few parameters applied at high resolution can dominate the runtime.

---

## Exercise: recompute for D = 24

The v0.1.0 checkpoint used `compressed_channels = 24`. The manifest says it has
**5,205,538** parameters. Can you get there?

Only four things change. Work them out before reading the answer.

<details>
<summary>Hint — the four things</summary>

1. the compressor's convolution and BatchNorm,
2. stage 1's patch embedding (its input is now 24 channels, not 32),
3. the decompressor,
4. the decoder's final convolution, because `out_channels * 16` is now
   `24 * 16 = 384` rather than 512.

</details>

<details>
<summary>Full answer</summary>

**1. Compressor**

```
Conv2d(165, 24, k=1) : 165*24 + 24 = 3,960 + 24 = 3,984
BatchNorm2d(24)      : 2 * 24                   =    48
                                                  -----
                                                  4,032      (was 5,376)
```

**2. Stage 1 patch embedding**

```
Conv2d(24, 32, k=4) : 24*32*16 + 32 = 12,288 + 32 = 12,320
LayerNorm(32)                                     =     64
                                                    ------
                                                    12,384   (was 16,480)
```

**3. Decompressor**

```
Conv2d(24, 165, k=1) : 24*165 + 165 = 3,960 + 165 = 4,125    (was 5,445)
```

**4. Decoder's final convolution**

```
Conv2d(256, 384, k=3) : 256*384*9 + 384 = 884,736 + 384 = 885,120
                                                          (was 1,180,160)
```

**The differences**

```
compressor        4,032 - 5,376     =    -1,344
patch embed 1    12,384 - 16,480    =    -4,096
decompressor      4,125 - 5,445     =    -1,320
decoder refine 2  885,120 - 1,180,160 = -295,040
                                        ---------
                          total         -301,800
```

**New total**

```
5,506,629 - 301,800 = 5,204,829 trainable parameters
```

**Plus buffers** — note the BatchNorm is now 24 channels:

```
normalisation buffers : 660
BatchNorm running     : 24 + 24 + 1 = 49
                        ---
                        709

5,204,829 + 709 = 5,205,538
```

**Exactly the manifest's figure for v0.1.0.**

Notice where the saving came from: 98% of it is the decoder's last convolution,
not the compressor itself. Shrinking `D` mostly shrinks the *decoder*.

</details>

---

## Common confusions

**"Does `num_heads` change the parameter count?"**
No. Multi-head attention splits the same matrices into groups; it does not add
any. That is why the table above never uses it.

**"Why is the parameter count independent of patch size?"**
Because convolutions and linear layers are applied at every position with the
same weights. A bigger patch means more work, not more parameters.

**"Is a 5.5M-parameter model big or small?"**
Small, by modern standards. ResNet-50 has 25M. A large language model has
hundreds of billions. This is deliberately a compact model — it has to run in a
worker container, sometimes on a CPU.

---

## Check yourself

1. How many parameters does `Conv2d(64, 128, kernel_size=3)` have?
2. How many does the depthwise version, `groups=64`, `Conv2d(64, 64, k=3)`,
   have?
3. Which single stage holds the most parameters, and why is that surprising?
4. Reconcile 5,506,629 with the manifest's 5,507,354.
5. Why does a layer with few parameters sometimes dominate the runtime?

<details>
<summary>Answers</summary>

1. `64 * 128 * 9 + 128 = 73,728 + 128 = 73,856`.
2. `64 * 9 + 64 = 576 + 64 = 640`. About 115x fewer.
3. Stage 4, with 1.6 million — surprising because it processes only 16 tokens.
   Parameters follow channel width, not token count.
4. The difference of 725 is buffers: 4 x 165 = 660 normalisation values, plus 65
   BatchNorm values (running mean 32, running variance 32, batch counter 1).
5. Because compute scales with the number of positions the layer is applied to.
   Stage 1 has few parameters but runs on 128x128 inputs, so its multiply-add
   count is enormous.

</details>

---

**Next:** where the training pictures come from, in
[27-training-data.md](27-training-data.md)
