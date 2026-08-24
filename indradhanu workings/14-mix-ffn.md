# 14 · Mix-FFN: the position-aware feed-forward network

> **The one thing this part teaches:** after tokens have consulted each other,
> each one needs private thinking time. Mix-FFN is that, plus a clever trick
> that removes the need for positional encodings.

**Source:**
[`app/foundation_models/components/mix_ffn.py`](../app/foundation_models/components/mix_ffn.py)

---

## Why a block needs more than attention

Attention (parts 12 and 13) moves information **between** tokens. But look at
what it actually computes: three matrix multiplies, a softmax, and a weighted
average. The only nonlinearity in the whole thing is the softmax, and that
operates on the weights, not on the content.

So attention alone cannot do much per-token processing. It gathers, it does not
think.

Every transformer therefore pairs attention with a small network that processes
each token **on its own**:

```
Linear(C -> C*E)  ->  some nonlinearity  ->  Linear(C*E -> C)
```

`E` is the "expansion ratio", 4 in this model. The middle is deliberately wider
than the ends.

> **Why go wider in the middle?** A wider intermediate layer gives the network
> more room to compute something genuinely nonlinear before squeezing back down.
> Think of it as unfolding a problem into more dimensions, solving it there, and
> folding the answer back up. Going straight from 32 to 32 through one
> nonlinearity is far more restrictive.

---

## SegFormer's twist: a convolution in the middle

Standard FFN:

```
Linear -> activation -> Linear
```

SegFormer's Mix-FFN:

```
Linear -> reshape to 2-D -> DepthwiseConv3x3 -> reshape to sequence -> GELU -> Linear
```

That 3x3 convolution mixes each token with its **eight spatial neighbours** on
the token grid.

### Why this replaces positional encoding

Without it, a transformer literally cannot tell where anything is. Shuffle the
tokens and attention gives you the same answers in a shuffled order — it has no
concept of "next to".

Standard transformers fix this by adding a learned position vector to every
token. SegFormer instead relies on this depthwise convolution: because it
operates on the actual 2-D grid, information naturally flows between spatially
adjacent tokens, and the network builds up a sense of layout.

That is why, as noted in part 12, there is no `pos_embed` anywhere in this
codebase.

---

## What "depthwise" means

```python
self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3,
                        padding=1, groups=hidden_dim)
```

The magic word is `groups=hidden_dim`.

In an **ordinary** convolution, every output channel is built from **all** input
channels. With 128 in and 128 out and a 3x3 kernel, that is
`128 x 128 x 9` weights.

In a **depthwise** convolution, each channel gets its own private 3x3 filter and
never looks at any other channel. That is `128 x 9` weights.

```
ordinary  : 128 * 128 * 9 + 128 = 147,584
depthwise : 128 *   9     + 128 =   1,280
```

**115 times cheaper.** And nothing is lost, because the `Linear` layers on
either side already do plenty of cross-channel mixing. The convolution's job is
purely spatial; let it do only that.

> **Analogy.** Ordinary convolution is a committee where every member consults
> every other member about every decision. Depthwise is 128 specialists each
> working alone on their own channel, with the mixing handled separately by the
> surrounding layers. Same outcome, far less meeting time.

The settings `stride=1, padding=1` keep the grid exactly the same size.
**Mix-FFN never changes the token count.** Tokens in equals tokens out, always.

---

## GELU

The nonlinearity used here is GELU — Gaussian Error Linear Unit.

Practically, treat it as a **smooth ReLU**:

```
ReLU:  flat zero for negatives, then a sharp corner, then a straight line
GELU:  near zero for very negative, a soft curve through zero, then near-linear
```

The softness matters for training: a sharp corner means the derivative jumps
abruptly, whereas GELU's derivative changes smoothly everywhere. Modern
transformers essentially all use it.

Note **where** it sits:

```
fc1 -> dwconv -> GELU -> fc2
```

After the convolution, before the final projection — that is, in the wide
`C*E`-dimensional space, where there are the most degrees of freedom for the
nonlinearity to work with.

---

## The sparse fallback — the same story as part 13

The depthwise convolution needs a real 2-D grid, so it needs `N == H * W`. After
tokens are hidden at stage 1, that is false.

The code handles it:

```python
if N == H * W:
    # standard path: reshape to 2-D, apply depthwise conv, reshape back
    ...
else:
    # sparse path: skip the depthwise conv entirely
    sequence_x = expanded_x
```

So during masked passes, stage 1's Mix-FFN is just a plain two-layer MLP with a
GELU in the middle. The docstring is honest about the trade-off:

> *"The positional mixing from dwconv is a nice-to-have, not essential — the
> attention in ESA already provides spatial context. Skipping dwconv for the
> ~512 visible tokens at Stage 1 is a minor quality tradeoff for correct
> handling of sparse token sets."*

Stages 2, 3 and 4 always take the standard path, because tokens are only ever
hidden at stage 1.

---

## Parameter count, worked

Stage 1: `C = 32`, `E = 4`, so `hidden = 32 * 4 = 128`.

```
fc1      Linear(32 -> 128)   : 32 * 128 + 128  =  4,096 + 128  = 4,224
dwconv   depthwise 3x3, 128ch: 128 * 9 + 128   =  1,152 + 128  = 1,280
fc2      Linear(128 -> 32)   : 128 * 32 + 32   =  4,096 + 32   = 4,128
                                                               -------
                                                    total        9,632
```

### Compare with attention at the same stage

| Stage | ESA | Mix-FFN | which dominates |
|---|---|---|---|
| 1 (`C=32`, `R=8`) | 69,856 | 9,632 | **attention**, because of the 8x8 reduction kernel |
| 4 (`C=256`, `R=1`) | 263,168 | 535,808 | **Mix-FFN**, because width dominates and there is no reduction conv |

The balance flips completely between the ends of the network. Part 26 lays the
whole budget out.

---

## Common confusions

**"Is Mix-FFN mixing tokens or mixing channels?"**
Both, in different places. `fc1` and `fc2` mix channels within each token. The
depthwise convolution mixes across nearby tokens, within each channel. Between
them, everything gets mixed — hence "Mix".

**"Does Mix-FFN reduce the number of tokens?"**
No. `stride=1, padding=1` preserves the grid exactly. Token count in = token
count out. Only patch embeddings change token count.

**"Why is the activation after the convolution rather than before?"**
So that the nonlinearity operates in the wide 128-dimensional space rather than
the narrow 32-dimensional one. More room to do something useful.

**"Is skipping the dwconv during masked passes a bug?"**
No — it is a deliberate, documented trade-off. Attention still provides spatial
context, and the alternative would be a crash.

---

## Check yourself

1. Why does a transformer block need a feed-forward network as well as
   attention?
2. What does `groups=hidden_dim` do, and how much does it save at
   `hidden_dim = 128`?
3. Where does this architecture's sense of position come from?
4. Compute Mix-FFN's parameters for stage 2 (`C = 64`, `E = 4`).
5. What happens to the depthwise convolution when tokens have been hidden?

<details>
<summary>Answers</summary>

1. Attention is nearly linear — it gathers information but does little per-token
   processing. The FFN provides the nonlinear per-token computation.
2. It makes the convolution depthwise: each channel gets its own filter with no
   cross-channel mixing. `128*9 + 128 = 1,280` instead of `128*128*9 + 128 =
   147,584` — about 115x cheaper.
3. From the depthwise 3x3 convolution inside Mix-FFN, which operates on the 2-D
   token grid. There is no positional encoding.
4. `hidden = 256`. fc1: `64*256 + 256 = 16,640`. dwconv: `256*9 + 256 = 2,560`.
   fc2: `256*64 + 64 = 16,448`. Total **35,648**.
5. It is skipped entirely, because the surviving tokens cannot be arranged into
   a full 2-D grid. The layer becomes a plain MLP for that pass.

</details>

---

**Next:** wiring attention and Mix-FFN into one repeatable unit, in
[15-the-block.md](15-the-block.md)
