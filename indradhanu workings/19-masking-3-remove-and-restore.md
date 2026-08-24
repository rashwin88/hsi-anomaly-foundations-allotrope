# 19 · Masking, step 3: remove and restore

> **The one thing this part teaches:** hidden tokens are not zeroed out, they are
> physically deleted from the list — and then put back, as zeros, after the
> encoder has finished.

**Source:** `TokenMasking.remove_tokens` and `TokenMasking.restore_tokens` in
[`token_masking.py`](../app/foundation_models/components/token_masking.py)

---

## First, two operations you need

Almost all of this part is `gather` and `scatter`. They are simple, they are
inverses of each other, and they are worth ten minutes.

### gather = "read from the positions I name"

```
data    = [ a, b, c, d, e ]
indices = [ 3, 0, 4 ]

gather  ->  [ d, a, e ]
```

Position 3 is `d`, position 0 is `a`, position 4 is `e`. The output is as long
as the *index* list, not the data list.

### scatter = "write into the positions I name"

```
start   = [ 0, 0, 0, 0, 0 ]
indices = [ 3, 0, 4 ]
values  = [ d, a, e ]

scatter ->  [ a, 0, 0, d, e ]
```

Value `d` goes to position 3, `a` to position 0, `e` to position 4. Everything
else keeps its starting value.

> **They are exact inverses when you use the same index list.** Gather the
> tokens out, do something to them, scatter them back, and everything lands
> where it started. That is precisely the pairing used here.

---

## Why physically remove, rather than zeroing?

You could hide a token by setting it to zero and leaving it in the list. This
code goes further and deletes it. Two reasons.

### Reason 1 — absolutely zero leakage

A zeroed token still occupies a slot. It is still a key that attention can
attend to. Even a zero vector participates in the softmax and takes a slice of
the weight.

Delete it and it cannot influence anything, because it is not there. This is the
strongest possible version of the "never see the thing you are predicting" rule
from part 06.

### Reason 2 — it is cheaper

With `mask_ratio = 0.65`, stage 1 processes about 358 tokens instead of 1024.
Every block, every attention, every matrix multiply — all at a third of the size.

Masking makes MAE training *faster*, not slower. That is one of the reasons the
technique became popular.

---

## remove_tokens, line by line

```python
num_kept       = keep_mask.sum(dim=1).long()
max_kept       = num_kept.max().item()
sorted_indices = keep_mask.argsort(dim=1, descending=True)   # 1s first
gather_indices = sorted_indices[:, :max_kept]                # (B, max_kept)

gather_indices_expanded = gather_indices.unsqueeze(-1).expand(-1, -1, C)
kept_tokens = torch.gather(tokens, dim=1, index=gather_indices_expanded)
```

Same idea as part 18: sorting instead of branching, so it runs as one GPU
operation for the whole batch.

**Sort descending** puts all the 1s (keep) before all the 0s (remove). Then the
first `max_kept` entries of the sorted index list are exactly the positions we
want.

### Worked example

`B = 1`, `N = 6`, `C = 2` (two features per token), tokens `t0` through `t5`,
and the `keep_mask` we computed in part 18:

```
keep_mask = [0, 1, 1, 1, 0, 1]        keep 4, remove 2 (positions 0 and 4)
```

**Sort descending, keeping ties in index order:**

```
keep values:  1   1   1   1   0   0
from position: 1   2   3   5   0   4

sorted_indices = [1, 2, 3, 5, 0, 4]
```

**Count and truncate:**

```
num_kept = 0+1+1+1+0+1 = 4
max_kept = 4
gather_indices = first 4 of sorted_indices = [1, 2, 3, 5]
```

**Gather:**

```
kept_tokens = [ t1, t2, t3, t5 ]          shape (1, 4, 2)
```

The list went from 6 tokens to 4, and the two hidden ones are simply gone.

### The `expand(-1, -1, C)` step

`torch.gather` requires the index tensor to have the same shape as the desired
output. Our indices are `(B, max_kept)` but the output must be
`(B, max_kept, C)` — we want to pull **whole token vectors**, not single
numbers.

So each index is repeated `C` times:

```
gather_indices          = [[1, 2, 3, 5]]
gather_indices_expanded = [[[1,1], [2,2], [3,3], [5,5]]]
                             ^^^^ index 1, repeated for both features
```

(`expand` does not actually copy memory; it creates a view that pretends to.
`-1` means "leave this dimension as it is".)

### The documented wrinkle: `max_kept`

`max_kept` is the **maximum across the whole batch**. But different batch items
may have different numbers of valid tokens, and therefore different numbers of
kept tokens.

If item A keeps 400 tokens and item B keeps 380, then `max_kept = 400` and item
B's gather pulls 400 indices — the last 20 of which come from the *removed* part
of its sorted list. So 20 tokens that should have been hidden slip back in.

The source is honest about it:

> *"If not, shorter items get a few extra tokens from the sort tail (minor
> leakage)."*

Why it is tolerable in practice:

- the mask ratio is uniform, so counts differ only when validity differs;
- the trainer's 40%-valid patch filter (part 23) keeps batches homogeneous;
- it affects a handful of tokens out of a thousand, occasionally.

Worth knowing about. Not worth losing sleep over.

---

## restore_tokens, line by line

After the blocks have processed the surviving tokens, they have to go back to
their original slots — otherwise you cannot reshape the list into a 2-D grid.

```python
full_tokens     = torch.zeros(B, N, C, device=device)
scatter_indices = gather_indices.unsqueeze(-1).expand(-1, -1, C)
full_tokens.scatter_(1, scatter_indices, kept_tokens)
```

Note it starts from **zeros**, and only the kept positions get written. Whatever
is not written stays zero.

### Worked example, continuing

Encoded tokens `t1'`, `t2'`, `t3'`, `t5'` (the prime marks "after the blocks"),
with the same indices `[1, 2, 3, 5]`:

```
start:   [  0 ,  0 ,  0 ,  0 ,  0 ,  0  ]

write:   position 1 <- t1'
         position 2 <- t2'
         position 3 <- t3'
         position 5 <- t5'

result:  [  0 , t1', t2', t3',  0 , t5' ]
            ^^^                 ^^^
            positions 0 and 4 — the hidden ones — remain exactly zero
```

Then the encoder reshapes:

```python
x = x.transpose(1, 2).reshape(B_cur, C_cur, H, W)
```

and stage 1's feature map `F1` is a 32x32 grid with zeroed holes exactly where
the hidden tokens were.

---

## Who fills the holes in?

**Nobody, at stage 1.** F1 keeps its holes. That is not an oversight; here is
how the gaps get resolved:

**1. The pyramid dilutes them.** Stage 2's patch embedding is a stride-2
convolution with a 3x3 window, so each stage-2 token pools over a 3x3
neighbourhood of stage-1 tokens. Most such neighbourhoods contain at least one
survivor. So stage 2's map is fully populated, and stages 3 and 4 more so.

**2. The decoder fuses all four scales.** When it predicts what belongs at a
hidden position, it draws on:

- the surviving stage-1 tokens nearby,
- the fully-populated F2, F3 and F4 covering the same ground.

That is why the model can be graded at hidden positions at all — and it is why
the multi-scale design in part 16 is not a luxury.

---

## The full round trip

```
(B, 1024, 32)                all tokens from the patch embedding
      |
      | remove_tokens(keep_mask)
      v
(B, 358, 32)                 only the survivors
      |
      | 2 x SegFormerBlock, then LayerNorm
      v
(B, 358, 32)                 processed
      |
      | restore_tokens(gather_indices, N=1024, C=32)
      v
(B, 1024, 32)                back in place, zeros in the gaps
      |
      | transpose + reshape
      v
(B, 32, 32, 32)              F1, ready for the decoder
```

---

## Common confusions

**"Does the encoder know a token is missing?"**
During the blocks, no — the token is simply absent. After restoration, the zeros
are a visible signal that something was there, but nothing says *what*.

**"Why not use a learned 'mask token' like the original MAE paper?"**
Because the decoder here does spatial fusion across four scales rather than
sequence modelling. Zeros are enough. Simpler code, one fewer parameter tensor.

**"Is `gather_indices` needed after `remove_tokens`?"**
Yes, absolutely. It is the only record of where each surviving token came from.
Lose it and you cannot restore. That is why `remove_tokens` returns it and the
encoder holds on to it across the blocks.

**"Does this happen at every stage?"**
No. Stage 1 only. Note the `i == 0` guard on both calls in the encoder loop.

---

## Check yourself

1. Explain `gather` and `scatter` in one sentence each.
2. Give the two reasons hidden tokens are deleted rather than zeroed.
3. `keep_mask = [1, 0, 1, 1]`, tokens `t0..t3`. What does `remove_tokens`
   return?
4. Continuing that, after the blocks produce `t0', t2', t3'`, what does
   `restore_tokens` produce?
5. What is the `max_kept` leakage issue, and why is it acceptable?

<details>
<summary>Answers</summary>

1. `gather` reads values from the positions you name. `scatter` writes values
   into the positions you name.
2. Zero leakage — a deleted token cannot be attended to at all; and speed — the
   blocks process about a third as many tokens.
3. `sorted_indices` descending is `[0, 2, 3, 1]`; `max_kept = 3`; so
   `gather_indices = [0, 2, 3]` and `kept_tokens = [t0, t2, t3]`.
4. `[t0', 0, t2', t3']` — position 1 stays zero.
5. `max_kept` is the batch maximum, so a batch item with fewer kept tokens pulls
   a few extra from the tail of its sort order — tokens that should have been
   hidden. It is acceptable because the 40% validity filter keeps batches
   homogeneous, so it affects very few tokens.

</details>

---

**Next:** the other mask in the system, in [20-mask-erosion.md](20-mask-erosion.md)
