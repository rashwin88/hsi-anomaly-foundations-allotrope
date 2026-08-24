# 16 · The four-stage encoder

> **The one thing this part teaches:** the encoder looks at the picture at four
> zoom levels and hands all four to the decoder — and it hides tokens at the
> first zoom level only.

**Source:**
[`app/foundation_models/components/seg_former_encoder.py`](../app/foundation_models/components/seg_former_encoder.py).
It is 261 lines, over half of them explanatory comments. Read it end to end
after this part; it will feel easy.

---

## What "hierarchical" means

A plain Vision Transformer keeps one resolution from start to finish. SegFormer
does something different: four stages, each at half the resolution and roughly
double the width of the one before.

```
input from the compressor:  (B, 32, 128, 128)

  Stage 1  ->  F1 : (B,  32, 32, 32)     quarter scale    fine texture
  Stage 2  ->  F2 : (B,  64, 16, 16)     eighth scale
  Stage 3  ->  F3 : (B, 160,  8,  8)     sixteenth scale
  Stage 4  ->  F4 : (B, 256,  4,  4)     thirty-second    global context
```

Two things move in opposite directions:

- **spatial detail decreases** — 32x32 down to 4x4,
- **channel width increases** — 32 up to 256.

That is the classic pyramid design. Early stages know *where* things are but
little about *what* they are. Late stages know a lot about *what* but have
almost no spatial precision.

### Why this matters for anomaly detection specifically

Anomalies come in wildly different sizes:

| Anomaly | Which stage sees it best |
|---|---|
| single odd pixel | F1 — it is the only one with the resolution |
| a 20-pixel patch of unusual material | F2, F3 |
| a 200-pixel region that does not belong | F4 |

A single-resolution model has to pick one of those. The encoder returns **all
four**, and the decoder fuses them (part 21). Nothing has to be chosen in
advance.

> **Analogy.** Reading a map at four zoom levels at once: street level for
> individual buildings, city level for districts, country level for the region.
> You would not survey unfamiliar terrain with only one of them.

---

## One stage, in code

Here is the loop body, which runs four times:

```python
for i in range(self.num_stages):
    x, H, W = self.patch_embeds[i](x)        # spatial -> tokens
    B, N, C = x.shape

    gather_indices = None
    if i == 0 and keep_mask is not None:     # hide tokens, stage 1 only
        x, gather_indices = TokenMasking.remove_tokens(x, keep_mask)

    for block in self.blocks[i]:             # 2 x SegFormerBlock
        x = block(x, H, W)

    x = self.norms[i](x)                     # stage output LayerNorm

    if i == 0 and gather_indices is not None:
        x = TokenMasking.restore_tokens(x, gather_indices, N, C)

    B_cur, N_cur, C_cur = x.shape
    x = x.transpose(1, 2).reshape(B_cur, C_cur, H, W)   # tokens -> spatial
    features.append(x)
```

Six steps: **embed, (maybe hide), blocks, normalise, (maybe restore), reshape.**

Note the last line does double duty. The reshaped tensor is both:

1. this stage's feature map, saved for the decoder, and
2. the next stage's input.

---

## The configuration

This is the SegFormer-B0 setup, with heads adjusted:

| Stage | `embed_dim` | `num_heads` | `head_dim` | `R` | `num_blocks` | resolution | tokens |
|---|---|---|---|---|---|---|---|
| 1 | 32 | 2 | 16 | 8 | 2 | H/4 | 1024 |
| 2 | 64 | 2 | 32 | 4 | 2 | H/8 | 256 |
| 3 | 160 | 5 | 32 | 2 | 2 | H/16 | 64 |
| 4 | 256 | 8 | 32 | 1 | 2 | H/32 | 16 |

(`num_heads = [2, 2, 5, 8]` in v0.2.0; v0.1.0 used `[1, 2, 5, 8]`. Nothing else
differs.)

---

## Hiding happens at stage 1 only

This surprises everybody, so here is the reasoning straight from the source:

> *"Token removal only happens at Stage 1. After scattering back and reshaping to
> 2D, Stage 2's OPE (Conv2d stride=2) pools over 3x3 regions, so most Stage 2
> tokens contain at least some real information even with 50% masking. Stages
> 2-4 process ALL tokens — the information loss from masking naturally dilutes
> through the spatial reduction."*

In plainer terms:

**The holes are punched once, at the finest resolution, and then the pyramid
blurs them out by itself.**

Stage 2's patch embedding is a stride-2 convolution with a 3x3 window. Each
stage-2 token therefore looks at a 3x3 neighbourhood of stage-1 tokens. With 65%
of stage-1 tokens hidden, most 3x3 neighbourhoods still contain at least one
survivor. So stage 2's tokens carry partial information everywhere, and by
stages 3 and 4 the gaps have essentially dissolved.

Punching fresh holes at every stage would be redundant *and* would starve the
coarse stages of any signal at all, which is exactly where the model gets its
"what usually goes here?" knowledge from.

---

## The complete masked walk-through

Two patches (`B = 2`), 128x128, and for readability let us use 50% hiding.

```
Stage 1:
   patch embed   (2, 32, 128, 128)  ->  (2, 1024, 32)     all 1024 tokens
   remove        keep_mask has ~512 ones -> (2, 512, 32)  hidden ones deleted
   2 x block     (2, 512, 32) -> (2, 512, 32)             encoder never sees the targets
   LayerNorm     (2, 512, 32)
   restore       -> (2, 1024, 32)                         zeros at hidden slots
   reshape       -> (2, 32, 32, 32)   = F1                a grid with holes

Stage 2 (no hiding):
   patch embed   (2, 32, 32, 32) -> (2, 256, 64)          stride-2 conv pools over the holes
   2 x block     -> (2, 256, 64)
   reshape       -> (2, 64, 16, 16)   = F2                no holes any more

Stage 3:
   patch embed   -> (2, 64, 160)
   2 x block
   reshape       -> (2, 160, 8, 8)    = F3

Stage 4:
   patch embed   -> (2, 16, 256)
   2 x block
   reshape       -> (2, 256, 4, 4)    = F4

returns [F1, F2, F3, F4]
```

---

## Three things that look like bugs and are not

### 1. `H` and `W` keep referring to the full grid

After tokens are removed there are only ~512 of them, but `H` and `W` are still
32 and 32.

That is correct and necessary. `H` and `W` are needed for:

- the restore step (to know how big the full grid is), and
- the final reshape.

Meanwhile ESA and Mix-FFN notice that `N != H * W` and take their fallback paths
(parts 13 and 14). The mismatch is *how they detect masking*.

### 2. Hidden positions become exactly zero, not a special "mask token"

Classic MAE papers insert a **learned mask embedding** at hidden positions — a
vector the model trains specifically to mean "something was here".

This model just writes zeros:

```python
full_tokens = torch.zeros(B, N, C, device=device)
full_tokens.scatter_(1, scatter_indices, kept_tokens)
```

Simpler, and it works, because the decoder's job is spatial fusion rather than
sequence modelling. It figures out what goes in the gaps from the surrounding
stage-1 tokens and from the fully-populated stages 2 to 4.

### 3. F1 keeps its holes all the way to the decoder

Nothing fills them in at stage 1. F1 genuinely arrives at the decoder with
zeroed patches.

That is fine — the decoder receives four feature maps and only one of them has
holes. The other three cover the same ground at coarser resolution.

---

## What the encoder returns

```python
return features       # [F1, F2, F3, F4]
```

A plain Python list of four tensors, at four resolutions, with four channel
widths. That is the complete interface between the encoder and the decoder.

---

## Common confusions

**"Do stages 2–4 see the original image?"**
No. Stage 2 sees stage 1's output grid, stage 3 sees stage 2's, and so on. Only
stage 1 ever touches the compressed picture.

**"Is the encoder a bottleneck like in a normal autoencoder?"**
Sort of, but a hierarchical one. There is no single narrow layer; instead there
are four representations, and the coarsest is very compressed indeed (4x4).

**"Why does channel width go up as resolution goes down?"**
To keep roughly constant information capacity per stage while shifting from
"where" to "what". It is the standard convolutional-backbone pattern, inherited.

**"Could I use a patch size other than 128?"**
Yes, provided it divides cleanly by 32 (four halvings after the initial /4). The
resolver offers 64, 128 and 256.

---

## Check yourself

1. Name the four stage resolutions and channel widths for a 128x128 patch.
2. Why is masking applied only at stage 1?
3. What are the six steps of one stage's loop body?
4. What sits at a hidden token's position after `restore_tokens`, and how does
   that differ from classic MAE?
5. Why do `H` and `W` still describe the full grid after tokens are removed?

<details>
<summary>Answers</summary>

1. F1 `(32, 32, 32)`, F2 `(64, 16, 16)`, F3 `(160, 8, 8)`, F4 `(256, 4, 4)` —
   channels first, then spatial size.
2. Because the later stages' pooling naturally dilutes the holes; masking again
   would be redundant and would starve the coarse stages of signal.
3. Patch embed; maybe remove tokens; run the blocks; LayerNorm; maybe restore
   tokens; reshape to 2-D.
4. Exactly zeros. Classic MAE inserts a learned "mask token" embedding instead;
   this model does not bother, because the decoder infers the gaps from context
   and from the coarser stages.
5. Because they are needed to restore the full grid and to reshape at the end —
   and because the mismatch `N != H*W` is precisely how ESA and Mix-FFN detect
   that masking is active.

</details>

---

**Next:** the masking machinery in detail, starting with
[17-masking-1-token-validity.md](17-masking-1-token-validity.md)
