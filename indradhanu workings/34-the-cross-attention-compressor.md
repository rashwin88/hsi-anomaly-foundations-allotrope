# 34 - The cross-attention compressor, assembled and counted

> **The one thing this part teaches:** 32 learned questions are put to a pixel's band tokens,
> each answer becomes one output channel, and the whole thing costs almost exactly as many
> parameters as the 1x1 convolution it replaces.

## The data flow, for one pixel

```
spectrum           (202,)          reflectance, SpectralEarth
   |  tokenise (part 33)
band tokens        (202, 32)       value stretch + wavelength tag
   |  K and V projections
keys, values       (202, 32) each
                                    32 learned queries (32, 32) --> Q projection --> (32, 32)
   |  attention: 32 queries x 202 keys, 4 heads
answers            (32, 32)        one 32-number answer per query
   |  output projection, then Linear(32, 1) per answer
channels           (32,)           the compressed pixel
```

The output is 32 numbers per pixel - the same shape the current compressor hands the SegFormer
encoder (part 10). **Everything downstream of the compressor stays as it is.** That is a
deliberate constraint: change one thing, measure it.

For a whole patch, the 128 x 128 = 16,384 pixels are treated as a batch of independent
pixels:

| tensor | shape | note |
|---|---|---|
| input | `(B, 202, 128, 128)` | B patches |
| reshaped to pixels | `(B*16384, 202)` | each row one spectrum |
| tokens | `(B*16384, 202, 32)` | |
| keys, values | `(B*16384, 4, 202, 8)` | 4 heads x 8 = 32 |
| queries | `(B*16384, 4, 32, 8)` | same learned queries for every pixel |
| answers | `(B*16384, 32, 32)` | |
| channels | `(B*16384, 32)` -> `(B, 32, 128, 128)` | back to an image |

## Predict the parameter count first

Do this on paper before reading the code. Use `parameters = outputs x inputs + outputs` for
each linear layer (part 08, section 4), with `d = 32`, `D = 32`.

```
value stretch     Linear(1, 32)       1*32 + 32        =    64
learned queries   table (32, 32)      32*32            = 1,024
Q projection      Linear(32, 32)      32*32 + 32       = 1,056
K projection      Linear(32, 32)      32*32 + 32       = 1,056
V projection      Linear(32, 32)      32*32 + 32       = 1,056
output proj       Linear(32, 32)      32*32 + 32       = 1,056
to one channel    Linear(32, 1)       32*1 + 1         =    33
                                                         -----
total                                                    5,345
```

Compare with part 10:

```
today:        Conv2d(165 -> 32, k=1) + BatchNorm2d(32)   = 5,376
next:         cross-attention compressor                  = 5,345
```

Thirty-one parameters fewer. The new compressor is **not bigger**; it spends the same budget
differently - on questions and projections instead of one fixed recipe per output. Note also
what does *not* appear: the number of bands. Nothing in the count is 202, 224 or 239. That is
the sensor independence of part 33 showing up in the arithmetic.

## The code

This is a study prototype, written to verify this course's numbers. It is not in the
repository's model code.

```python
class SpectralCrossAttentionCompressor(nn.Module):
    def __init__(self, D=32, d=32, heads=4):
        super().__init__()
        self.D, self.d, self.heads = D, d, heads
        self.value_embed = nn.Linear(1, d)                       # part 33
        self.queries = nn.Parameter(torch.randn(D, d) * 0.02)    # the 32 questions
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.to_channel = nn.Linear(d, 1)

    def forward(self, x, wl_nm, band_valid=None):
        # x: (P, NB) reflectance, P pixels   wl_nm: (NB,)   band_valid: (P, NB) bool or None
        P, NB = x.shape
        tok = self.value_embed(x[..., None]) + wavelength_encoding(wl_nm, self.d)[None]
        # tok: (P, NB, d)
        k = self.k(tok).view(P, NB, self.heads, -1).transpose(1, 2)   # (P, h, NB, d/h)
        v = self.v(tok).view(P, NB, self.heads, -1).transpose(1, 2)   # (P, h, NB, d/h)
        q = self.q(self.queries).view(self.D, self.heads, -1).transpose(0, 1)  # (h, D, d/h)
        q = q[None].expand(P, -1, -1, -1)                             # (P, h, D, d/h)
        mask = None if band_valid is None else band_valid[:, None, None, :]   # part 35
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # (P, h, D, d/h)
        out = self.o(out.transpose(1, 2).reshape(P, self.D, self.d))   # (P, D, d)
        return self.to_channel(out).squeeze(-1)                        # (P, D)
```

Run on 16 pixels of the real 202-band SpectralEarth wavelength table
(`app/constants/spectralearth_bands.json`):

```
cross-attention compressor params: 5345
  value_embed 64, queries 1024, q 1056, k 1056, v 1056, o 1056, to_channel 33
forward: (16, 202) -> (16, 32)
```

Your hand count should match line for line.

## The 1x1 convolution is a special case

Write both as "output channel c of one pixel":

```
1x1 conv:     y_c = sum over bands b of   W[c, b]          * x_b
attention:    y_c = sum over bands b of   alpha[c, b](x)   * value(x_b, wavelength_b)
```

In the convolution the weight `W[c, b]` is a fixed number. In attention the weight
`alpha[c, b](x)` is a softmax over scores that depend on the whole spectrum `x`. Freeze the
alphas to constants and attention collapses back into a (more complicated) fixed weighted
sum. **The cross-attention compressor is the 1x1 convolution with data-dependent weights** -
that is the entire idea, and why it answers part 31's first requirement.

## Common confusions

**"Each query gives a 32-number answer; why do we need `to_channel`?"** The encoder expects
32 *numbers* per pixel, not 32 vectors. `Linear(32, 1)` turns each answer vector into one
number, and it is shared by all 32 queries (33 parameters, not 32 x 33).

**"Why 4 heads?"** Same reason as part 12: four groups of 8 numbers can learn four different
ways of scoring relevance. It changes no parameter count - the projections are 32x32 either
way.

**"Where did the BatchNorm go?"** The current compressor has one to steady the encoder's
input distribution (part 10, "the asymmetry"). Whether the new one needs a normalisation
layer is an open design question; leaving it out here keeps the count comparison honest.
Expect it to come back as a LayerNorm in the real design.

**"Does every pixel get its own queries?"** No. All 16,384 pixels share the same 1,024-number
query table. Only the keys and values differ per pixel.

## Check yourself

1. In one sentence, what does each of the 32 queries produce?
2. Recompute the parameter count for `d = 16`, `D = 32` (16 numbers per token, still 32
   output channels). Show each line.
3. Why does the band count (202, 224, 239) never appear in the parameter count?
4. A batch of 4 patches goes in. How many pixels does the compressor process as independent
   rows?
5. In the formula `y_c = sum alpha[c, b](x) * value(...)`, what would you have to freeze to get
   back a fixed recipe?

<details>
<summary>Answers</summary>

1. One output channel of the compressed pixel: a relevance-weighted blend of the band tokens,
   reduced to a single number.
2. Value stretch `1*16 + 16 = 32`; queries `32*16 = 512`; Q, K, V, O each `16*16 + 16 = 272`,
   so `4*272 = 1,088`; to-channel `16*1 + 1 = 17`. Total `32 + 512 + 1,088 + 17 = 1,649`.
3. Every layer acts on one token at a time with shared weights, or on the fixed query table;
   no layer has a weight per band. The band count sets how many tokens there are, not how
   many weights.
4. `4 * 128 * 128 = 4 * 16,384 = 65,536`.
5. The attention weights `alpha[c, b]` - make them constants instead of a softmax of
   data-dependent scores.

</details>

---

**Next:** part 35 - skipping a dead band, exactly.
