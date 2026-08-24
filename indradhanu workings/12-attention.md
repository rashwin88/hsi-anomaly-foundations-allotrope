# 12 · Attention, from first principles

> **The one thing this part teaches:** attention is a similarity-weighted average.
> Each token asks every other token "how relevant are you to me?", and updates
> itself with a blend of their answers.

This part explains ordinary self-attention. Part 13 explains the single
modification SegFormer makes. If you have never met attention before, go slowly
and do the arithmetic — it is only addition and multiplication.

---

## The question attention answers

After part 11, each token holds a summary of its own 4x4 block of the picture.
But a block's *meaning* depends on its surroundings.

A dark block is unremarkable in the middle of a lake. The identical dark block
is very interesting in the middle of a bright desert.

So every token needs a way to **look at other tokens and revise itself**.
Attention is that mechanism, and crucially, *what it pays attention to is
learned* rather than fixed.

---

## Three roles: query, key, value

From each token's vector, three new vectors are computed, using three separate
learned matrices:

```python
self.q = nn.Linear(embed_dim, embed_dim)
self.k = nn.Linear(embed_dim, embed_dim)
self.v = nn.Linear(embed_dim, embed_dim)
```

(`nn.Linear` is the matrix multiply from part 08.)

The standard metaphor — genuinely helpful, so use it:

| Role | Think of it as | Analogy |
|---|---|---|
| **query** | what this token is looking for | your search box text |
| **key** | what this token advertises about itself | a document's title |
| **value** | what this token hands over if consulted | the document's contents |

Every token produces all three. Every token is simultaneously a searcher and a
searchable document.

---

## The four steps

```
1.  scores  = Q @ K^T * (1 / sqrt(head_dim))
2.  weights = softmax(scores, dim=-1)
3.  out     = weights @ V
4.  out     = proj(out)
```

In words:

1. Compare every query against every key. High score = relevant.
2. Convert scores to proportions that add up to 1.
3. Take a weighted average of the values, using those proportions.
4. One final matrix multiply to mix the result.

That is all attention is. **A similarity-weighted average of other tokens'
values.**

> **Notation.** `@` is matrix multiplication in Python. `K^T` means K
> transposed (rows and columns swapped), which is what makes the multiplication
> compare every query against every key.

---

## Worked example, entirely by hand

To keep the numbers readable: **3 tokens**, each with **2 features**, **1
attention head**, and we pretend the three matrices are all the identity, so
that Q, K and V are simply the tokens themselves.

```
x1 = [1, 0]
x2 = [0, 1]
x3 = [1, 1]
```

`head_dim = 2`, so `scale = 1/sqrt(2) = 0.7071`.

### Step 1 — scores for token 1

Token 1's query is `[1, 0]`. Dot it against each key (part 08, section 1):

```
q . k1 = 1*1 + 0*0 = 1
q . k2 = 1*0 + 0*1 = 0
q . k3 = 1*1 + 0*1 = 1
```

Multiply each by the scale:

```
raw:    [1, 0, 1]
scaled: [0.7071, 0.0000, 0.7071]
```

Interpretation so far: token 1 finds tokens 1 and 3 relevant (both have a strong
first feature, which is what token 1 is asking about), and token 2 irrelevant.

### Step 2 — softmax

```
exp(0.7071) = 2.0281
exp(0.0000) = 1.0000
exp(0.7071) = 2.0281
                ------
sum:            5.0562

weights: 2.0281/5.0562 = 0.4011
         1.0000/5.0562 = 0.1978
         2.0281/5.0562 = 0.4011
```

Check they sum to 1: `0.4011 + 0.1978 + 0.4011 = 1.0000`. Good.

### Step 3 — weighted average of the values

```
out1 = 0.4011 * [1, 0]
     + 0.1978 * [0, 1]
     + 0.4011 * [1, 1]
```

Take it feature by feature.

First feature:

```
0.4011 * 1  = 0.4011
0.1978 * 0  = 0.0000
0.4011 * 1  = 0.4011
              ------
              0.8022
```

Second feature:

```
0.4011 * 0  = 0.0000
0.1978 * 1  = 0.1978
0.4011 * 1  = 0.4011
              ------
              0.5989
```

```
out1 = [0.8022, 0.5989]
```

### What just happened

Token 1 went in as `[1, 0]` and came out as `[0.80, 0.60]`.

It has absorbed context. Its second feature was zero and is now 0.60, because
its neighbours had a second feature and it decided they were worth listening to.

Repeat for tokens 2 and 3 and you have one complete attention layer. That is
genuinely it.

---

## Why divide by the square root

Step 1 has a `* (1 / sqrt(head_dim))` that we have not justified.

The problem: dot products get bigger as vectors get longer. With `head_dim = 64`
and typical values, raw scores would have a spread of about 8 — so you would
routinely see scores like 20 and -15.

Now feed that to softmax:

```
exp(20)  = 485,165,195
exp(-15) = 0.0000003
```

The largest score takes essentially **100%** of the weight and everything else
gets zero. Two bad consequences:

1. The token can only ever attend to exactly one other token — no blending.
2. During training, the near-zero weights produce near-zero gradients, so the
   other options never get a chance to improve. Learning stalls.

Dividing by `sqrt(head_dim)` keeps the spread of scores around 1, which keeps
softmax in its useful, gradual range.

The code puts it plainly:

```python
self.scale = self.head_dim ** -0.5
```

(`** -0.5` is "to the power of minus one half", which is one over the square
root.)

---

## Multiple heads

Instead of one attention over all `C` features, split the features into `h`
groups and run `h` independent attentions side by side, then glue the results
back together.

```
head_dim = embed_dim // num_heads
```

Stage 3 of this model: `embed_dim = 160`, `num_heads = 5`, so
`head_dim = 160 / 5 = 32`. Five separate attentions, each working with 32 of the
160 features.

**Why bother?** Because different heads can specialise. One might learn "attend
to my immediate neighbours". Another might learn "attend to anything with a
similar brightness anywhere in the patch". With a single head you get one
averaged compromise; with five you get five distinct behaviours.

> **Analogy.** One person reading a document and forming a single impression,
> versus five specialists reading it — a lawyer, an accountant, an engineer, a
> historian, an editor — each noticing different things. Then you combine their
> notes.

### The reshaping in code

```python
Q_x = Q_x.reshape(B, N, self.num_heads, self.head_dim)
Q_x = Q_x.transpose(1, 2)      # (B, num_heads, N, head_dim)
```

Read it as: split the feature axis into (head, feature-within-head), then move
the head axis up front so each head's data is contiguous.

Afterwards, the reverse, followed by the final projection:

```python
out = out.transpose(1, 2)
out = torch.reshape(out, (B, N, self.head_dim * self.num_heads))
out = self.proj(out)
```

That last `self.proj` matters: without it the heads would never interact, and
you would just have `h` separate models stapled together.

### The divisibility constraint

`embed_dim` must divide evenly by `num_heads`. Check this model's configuration:

| Stage | embed_dim | num_heads | head_dim | whole number? |
|---|---|---|---|---|
| 1 | 32 | 2 | 16 | yes |
| 2 | 64 | 2 | 32 | yes |
| 3 | 160 | 5 | 32 | yes |
| 4 | 256 | 8 | 32 | yes |

All fine. (Version 0.1.0 used `[1, 2, 5, 8]` heads, giving stage 1 a head_dim of
32 as well.)

---

## The cost problem, which sets up part 13

The score matrix has one entry for every (query, key) pair. With `N` tokens that
is `N x N` entries, per head, per batch item, per block.

At stage 1, `N = 1024`:

```
1024 x 1024 = 1,048,576 entries
```

Over a million numbers, for one head, in one block, for one patch. And stage 1
is also where the tensors are largest and the resolution finest.

This is the single reason part 13 exists.

---

## No positional encoding — really

In standard transformers, tokens carry no inherent sense of position, so a
"positional encoding" is added to tell them where they are.

SegFormer has none. There is no `pos_embed` parameter anywhere in this
codebase — go and grep for it.

Instead, position enters through a **depthwise convolution inside the
feed-forward network** (part 14), which mixes each token with its spatial
neighbours. Because that convolution operates on the 2-D grid, tokens implicitly
learn where they are relative to each other.

It is an elegant trick: the position information comes from the geometry of an
operation rather than from an extra learned table.

---

## Common confusions

**"Is attention the same as correlation?"**
Related but not identical. Correlation is a fixed statistical measure. Attention
uses *learned* projections before comparing, so it can learn what "relevant"
means for this task.

**"Does every token attend to every other token?"**
In plain attention, yes. In this model's efficient variant, queries are
full-resolution but keys and values are summarised first — part 13.

**"Why three separate matrices? Why not compare tokens directly?"**
Because "what I am looking for" and "what I advertise" are usually different
things. Separate projections let a token search for one thing while offering
another.

**"What does 'self' in self-attention mean?"**
That queries, keys and values all come from the *same* set of tokens. In
cross-attention (not used here) queries come from one sequence and keys/values
from another.

---

## Check yourself

1. Recite the four steps of attention.
2. In one sentence each, what are the query, key and value?
3. Redo the worked example for **token 2** (`q = [0, 1]`). What weights does it
   assign, and what is its output?
4. Why is the scale factor `1/sqrt(head_dim)` necessary?
5. Stage 4 has `embed_dim = 256` and `num_heads = 8`. What is `head_dim`, and
   how big is the score matrix if `N = 16`?

<details>
<summary>Answers</summary>

1. Score every query against every key (scaled); softmax into weights; weighted
   average of the values; final linear projection.
2. Query = what this token is looking for. Key = what it advertises. Value =
   what it contributes when attended to.
3. Scores: `q.k1 = 0`, `q.k2 = 1`, `q.k3 = 1`; scaled `[0, 0.7071, 0.7071]`;
   exponentials `[1.0000, 2.0281, 2.0281]`, sum 5.0562; weights
   `[0.1978, 0.4011, 0.4011]`. Output first feature:
   `0.1978*1 + 0.4011*0 + 0.4011*1 = 0.5989`. Second feature:
   `0.1978*0 + 0.4011*1 + 0.4011*1 = 0.8022`. So `out2 = [0.5989, 0.8022]` — the
   mirror image of token 1's result, as symmetry demands.
4. Without it, dot products grow with dimension, softmax saturates onto a single
   token, and gradients for the rest vanish so learning stalls.
5. `head_dim = 256 / 8 = 32`. Score matrix is `16 x 16 = 256` entries per head —
   trivially cheap, which is why stage 4 uses no reduction.

</details>

---

**Next:** making attention affordable, in
[13-efficient-self-attention.md](13-efficient-self-attention.md)
