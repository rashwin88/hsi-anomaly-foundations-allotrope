# 70 - Heavy in the product: the thick-cloud rule, parity, sixteen threads, and the FT08 leak

> **The one thing this part teaches:** on 2026-10-29 FTH3 epoch 5 replaced FT04 behind the codename
> Indradhanu V2 (`v2.1.0`), excluding thick cloud only, and the product path reproduced the
> evaluator's pAUC to four decimals (mean 0.1892); raising the worker from 2 to 16 threads made a
> PRISMA scene four times faster (Heavy 1819 s -> 454 s); and on 2026-10-31 a leak in FT08's
> evaluation, found while writing the textbook, was fixed - the re-run moved FT08's pAUC from 0.164 to
> 0.1638 and left the verdict "block masking does not help" exactly where it was.

> **Update 2026-10-31 (evening): native bands are now the band-filter default.** When this part was
> written, using V2 still meant setting `use_common_wavelength_grid: false` on the band filter by hand
> (part 61). Since the evening of 2026-10-31 every hyperspectral sensor default does that, and
> `anomaly_scoring` resamples a native cube to the 165-band grid for v0.2.0 (`_to_common_grid`), so V2
> runs from the UI with no band-filter setting and one action can run both models. V2 itself still
> refuses a 165-grid band filter. The same day, V2's stem got a second user, the spectral namer
> (part 71).

## Where we are

Part 69 ended with a release candidate, FTH3 epoch 5, and three things it needed before it could reach
the product: the thick-cloud rule, a manifest / resolver path for the progressive decoder, and a check
of the worker's CPU cost. All three were done on 2026-10-29 and Heavy shipped the same day. This part is
what shipping it involved, plus one correction to an older experiment (FT08, part 65).

| file | what it holds |
|---|---|
| `research/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/README.md` | the "Shipped" section: parity table, CPU timings |
| `.../FTH3-heavy-joint-more2/product_parity.py`, `cpu_timing.py` | the two scripts behind those tables |
| `backend/allotrope/foundation_models/resolver.py` | `thin_cloud_is_cloud` in `ModelCapabilities` |
| `backend/allotrope/action_types/_anomaly_scoring_run.py` | passes it to `segment_full_scene`; writes `cloud_rule` to `stats.json` |
| `app/foundation_models/inferencers/spectral_stem_inferencer.py` | `segment_full_scene(..., thin_cloud=...)`; `__init__`, the FT08 fix |
| `app/foundation_models/inferencers/segformer_mae_inferencer.py` | the two-pass checkerboard and how each pixel's pass is chosen |
| `docker/docker-compose.yml` | the worker's thread settings |
| `docs/lld/spectral-stem.md` ("Serving"), `docs/04-models.md`, `docs/09-known-issues.md` | the reference write-ups |
| `research/experiments/indradhanu-finetune/FT07-gated/README.md` | the final section: FT08 re-evaluated |
| `tests/test_foundation_models/test_gated_training.py` | the test that pins the fix |

> **Analogy.** A new specialist joins a clinic (part 61 used the same picture for FT04). Before her first
> patient three things are checked: she reads the same scans the same way the old consultant's notes
> say she should (parity); she is given a room with enough equipment that patients do not queue
> (threads); and the referral rule is changed for her one known blind spot - she over-reads haze as
> cloud, so only her "thick cloud" calls are acted on (the thick-cloud rule).

## What shipped

`docs/lld/spectral-stem.md`, "Serving": V2 is now "Indradhanu Heavy (FTH3 epoch 5, `v2.1.0`: 3x wide,
progressive decoder, 35.3 M params); FT04 (`v2.0.0`) is the manifest's first `alternatives` entry. Same
architecture key, so nothing in the resolver or worker branches on which one it is - `build_model` reads
`decoder_type`, `decoder_stage_dims` and the widths from the embedded `model_config`."

Two words from part 61, again. The **manifest** is the small JSON file next to the checkpoint that tells
the resolver which file to load and with what configuration; the **architecture key**
(`spectral_stem_finetune`) is what the resolver looks up to decide which inferencer class to build. Because
Heavy kept FT04's key and its manifest carries its own configuration, switching models was a change of
files, not of code. Rolling back is the same in reverse: the alternative is still listed.

**Predict the 35.3 M yourself.** Part 68 counted the *pretrained* Heavy at 33,884,194 parameters. The
fine-tuned model adds the segmentation head. Building FTH3's `model_config` on a CPU
(`research/textbook/tools/compute_heavy_product.py`) gives a `seg_head` of 1,431,120:

```
33,884,194 + 1,431,120 = 35,315,314 trainable parameters    -> the LLD's "35.3 M"
```

## The thick-cloud rule

Part 69 found that Heavy's **thin-cloud** output fires on haze: on PRISMA set4 it covered 75.4% of the
valid pixels and 209 of the 230 targets. The product's old rule zeroed "cloud or thin cloud", which
would have blanked most of a hazy scene. So V2 now zeroes **thick cloud only**.

The rule is one field, set once for the architecture, in `resolver.py`:

```python
# Whether the head's thin-cloud output counts as "cloud" for exclude_classes. False = thick cloud only: Indradhanu
# Heavy's thin-cloud output fires on haze (research/experiments/indradhanu-finetune/FTH1-heavy-joint/).
thin_cloud_is_cloud: bool = True              # the default in ModelCapabilities
...
exclude_classes=("cloud",),                   # spectral_stem_finetune's entry
thin_cloud_is_cloud=False,
```

The worker passes it on (`_anomaly_scoring_run.py`):

```python
class_map, _ = inferencer.segment_full_scene(model_cube, spatial_valid.astype(bool),
                                             thin_cloud=m.thin_cloud_is_cloud)
...
"cloud_rule": "cloud_or_thin" if m.thin_cloud_is_cloud else "thick"      # written to stats.json
```

and `segment_full_scene` turns the head's five probabilities into one class per pixel. The channels are
`prob[0]` water, `prob[1]` thick cloud, `prob[2]` thin cloud, `prob[3]` shadow, `prob[4]` snow:

```python
cls = np.zeros((H, W), np.int64)                 # 0 land
cls[prob[0] > 0.5] = 1                           # water
cls[prob[3] > 0.5] = 3                           # shadow
cls[prob[4] > 0.5] = 4                           # snow
cls[(np.maximum(prob[1], prob[2]) if thin_cloud else prob[1]) > 0.5] = 2     # cloud, written last
```

Each line overwrites the ones before it, so the last write wins: that is the priority
cloud > snow > shadow > water > land. Do one pixel by hand. Its probabilities are water 0.7, thick cloud
0.2, thin cloud 0.9, shadow 0.1, snow 0.0:

```
start                       class 0 (land)
water   0.7 > 0.5           class 1
shadow  0.1                 no change
snow    0.0                 no change
old rule:   max(0.2, 0.9) = 0.9 > 0.5   -> class 2, cloud: score set to 0
thick rule: 0.2 > 0.5 is false           -> stays class 1, water: scored (vessels on water are in scope)
```

The rule costs FT04 nothing: FT04 under the thick rule against FT04 under the old rule is
-0.0004 [-0.0015, +0.0015] (`docs/lld/spectral-stem.md`, from FTH2). The price is the one
`docs/09-known-issues.md` states: "Thin cirrus over a target is therefore scored, not masked."

## Parity: the product gives the evaluator's numbers

**Parity** is the check from part 61: run the product's own code path on the bench scenes and confirm it
gives the numbers the evaluator measured. `product_parity.py` goes resolver manifest -> `get_inferencer`
-> `read_bands` -> `predict` -> `compute_score(method="residual_mahalanobis")` ->
`segment_full_scene(thin_cloud=False)` -> zero the cloud, exactly the worker's order. From the FTH3 README:

| scene | thick cloud excluded | pAUC product | pAUC evaluator |
|---|---|---|---|
| set1 | 0.00% | 0.2617 | 0.26165 |
| set2 | 0.00% | 0.1874 | 0.18742 |
| set3 | 1.49% | 0.0861 | 0.08609 |
| set4 | 0.47% | 0.2217 | 0.22167 |

Average the product column:

```
0.2617 + 0.1874 = 0.4491
0.4491 + 0.0861 = 0.5352
0.5352 + 0.2217 = 0.7569
0.7569 / 4      = 0.1892
```

and the evaluator's: `0.26165 + 0.18742 + 0.08609 + 0.22167 = 0.75683`, `0.75683 / 4 = 0.1892`. The
per-scene gaps (0.00005, 0.00002, 0.00001, 0.00003) are the product column's rounding to four places.
**The product path and the evaluator agree to the digit.** Look at set4: 0.47% thick cloud, where the old
rule had called 75.4% of it cloud.

In objects, the comparison page the user was shown on 2026-10-29
(https://claude.ai/artifact/2dZFbGN6HhMBCx3kkZ8pQ4, "Heavy vs FT04 detections") counts the 909 labelled
targets on the four scenes caught in each model's top 1%: FT04 418, Heavy 449.

```
FT04   418 / 909 = 0.4598   -> 46.0%
Heavy  449 / 909 = 0.4939   -> 49.4%
gain   449 - 418 = 31 objects, 3.4 points
```

the same 46.0% and 49.4% as the FTH3 README's "caught top 1%" column. The page puts chance at 10.2%.

## Sixteen threads

The CPU check (`cpu_timing.py`, inside the `allotrope_worker` container, PRISMA set1,
1186 x 1196 pixels x 185 bands) first gave Heavy 1819 s and FT04 1058 s: about 30 and 18 minutes for one
scene. The worker was using **2 threads**. A **thread** is one line of work a program runs at a time; a
numerical library such as torch splits a big matrix product across as many threads as it is allowed.

Why 2? `backend/Dockerfile.worker` sets `OMP_NUM_THREADS=2` and its siblings (`MKL_`, `OPENBLAS_`,
`NUMEXPR_`), and `backend/allotrope_worker/__init__.py` does `_os.environ.setdefault(_var, "2")`. The
cap dates from crashes that turned out to be something else; the compose file's comment says so:

```yaml
# Thread counts. The SIGTERMs that once motivated a cap of 2 were Docker
# Desktop UI's restart button, not throttling (2026-05-11) - but the cap
# stayed in backend/Dockerfile.worker (ENV) and allotrope_worker/__init__.py
# (setdefault), so the worker ran torch on 2 threads until 2026-10-29
# (Indradhanu Heavy: ~30 min per PRISMA scene). Set here, they override
# both. 16 leaves headroom on the 24-core host. If kill cycles return,
# lower these before rebuilding.
OMP_NUM_THREADS: ${WORKER_THREADS:-16}
```

Why does the compose value win over both? Compose's `environment:` replaces the image's `ENV` when the
container starts. And `setdefault` only writes a variable that is **not already set**: compose has set it
to 16 before Python starts, so the `"2"` is never written. `${WORKER_THREADS:-16}` means "the
`WORKER_THREADS` variable if it is set, otherwise 16".

The timings with 16 threads, from the FTH3 README:

| | 2 threads | 16 threads | speed-up |
|---|---|---|---|
| Heavy total | 1819 s | 454 s | 1819 / 454 = 4.01 |
| Heavy predict | 1629 s | 402 s | 1629 / 402 = 4.05 |
| Heavy segment | 185 s | 47 s | 185 / 47 = 3.94 |
| Heavy score | 4 s | 4 s | 1.0 |
| FT04 total | 1058 s | 274 s | 1058 / 274 = 3.86 |
| FT04 predict | 920 s | 236 s | 920 / 236 = 3.90 |
| FT04 segment | 133 s | 32 s | 133 / 32 = 4.16 |

In minutes: `1819 / 60 = 30.3` down to `454 / 60 = 7.6` for Heavy, `1058 / 60 = 17.6` down to
`274 / 60 = 4.6` for FT04. Heavy still costs `454 / 274 = 1.66`, about 1.7 times FT04, as before.

Eight times the threads (`16 / 2 = 8`) bought four times the speed. Why not eight? Not every part of the
work splits across threads (the score step did not move at all), and threads share memory bandwidth. A
standard model of this is **Amdahl's law**: if a share `p` of the work splits perfectly and the rest does
not, the time on `n` threads is proportional to `(1 - p) + p / n`. This is a model, not a measurement;
use it only to get a feel. Going from 2 to 16 threads:

```
speed-up = ((1 - p) + p/2) / ((1 - p) + p/16)

set it to 4:   (1 - p/2) = 4 (1 - 15p/16)
               1 - 0.5 p = 4 - 3.75 p
               3.25 p    = 3
               p         = 0.923
```

So the measured speed-up behaves as if about 92% of the work were perfectly parallel. **The trap that
remains:** the Dockerfile and `__init__` still say 2, so anything that runs the worker outside compose
(a bare `docker run`, a test harness) is back on 2 threads and four times slower
(`docs/09-known-issues.md`).

## The FT08 leak

### What FT08 changed, briefly

FT08 (part 65) trained with **block masks**: tokens were hidden in 2 x 2-token blocks (8 x 8 pixels)
instead of one 4 x 4-pixel token at a time, so a target and its halo were hidden together. At inference
the product reconstructs a scene in two complementary checkerboard passes (part 29): pass 1 hides half
the tokens, pass 2 the other half, and each pixel's prediction is taken from the pass in which it was
hidden. For FT08 the checkerboard had to use 2 x 2-token cells too.

### Two choices that must agree

The parent class (`segformer_mae_inferencer.py`) makes two separate checkerboard choices:

```python
checker = TokenMasking.checkerboard_token_mask(H_tokens, W_tokens,
                                              cell_size=self.config.checkerboard_cell_size, ...)
pass1_pixels = self._token_mask_to_pixel_mask(1.0 - checker, ...)    # choice A: which pass each pixel comes from
pass2_pixels = self._token_mask_to_pixel_mask(checker, ...)
keep_mask_1 = self._checkerboard_keep_mask(H, W, mask, invert=False)  # choice B: which tokens each pass hides
keep_mask_2 = self._checkerboard_keep_mask(H, W, mask, invert=True)
...
reconstruction = x_hat_1 * pass1_pixels + x_hat_2 * pass2_pixels
```

Until 2026-10-31, `SpectralStemInferencer` overrode only choice B, `_checkerboard_keep_mask`, to use
cells of `mask_block` = 2. Choice A still read `config.checkerboard_cell_size`, which was 1. The masks
hid 2 x 2 blocks; the pass selection alternated token by token.

Draw it on a 4 x 4-token tile, with part 29's convention (pass 1 hides the cells whose row + column is
odd). Pass 1 hides the blocks marked `H` (cells of 2 tokens); choice A takes a token from pass 1 where it
shows `1` (cells of 1 token). A token is correct when it is taken from the pass that hid it:

```
pass 1 hides (cell 2)     choice A: from pass 1? (cell 1)     leak? (taken from a pass that saw it)
  . . H H                   . 1 . 1                              . x x .
  . . H H                   1 . 1 .                              x . . x
  H H . .                   . 1 . 1                              x . . x
  H H . .                   1 . 1 .                              . x x .
```

Token (0, 0): visible in pass 1, taken from pass 2, which hid it - correct. Token (0, 1): visible in
pass 1 and taken from pass 1 - a leak. Token (0, 2): hidden in pass 1 but taken from pass 2, where its
block was visible - a leak. Count the `x`s: 8 of 16. In every 2 x 2 block the
cell-1 pattern has two tokens of each colour, so **exactly half of each block's pixels were predicted by a
pass that could see them**, whichever way round the two patterns line up. The source comment calls it
"half of each block's pixels were taken from the pass where they were visible (found by the textbook
writers)" - the people turning these courses into a LaTeX book.

### The fix and its test

`SpectralStemInferencer.__init__` now copies `mask_block` into the configuration before the parent is
built, so both choices read the same number:

```python
def __init__(self, config):
    block = getattr(config.model_config_, "mask_block", 1)
    if block > config.checkerboard_cell_size:
        config = config.model_copy(update={"checkerboard_cell_size": block})
    super().__init__(config)
```

The old override of `_checkerboard_keep_mask` is gone. The test
`test_block_inference_never_predicts_a_pixel_from_a_pass_that_saw_it` builds a tiny model with
`mask_block=2`, adds 0.4 to every band of one 8 x 8 block (`y[:, :, 8:16, 16:24] += 0.4`) and asserts that
the predictions inside that block do not change. If any of those pixels were predicted by a pass that
could see the block, changing the block would change them. The FT07 README says the test "fails without
the fix".

### Why the verdict did not change

Re-run on 2026-10-31 (`configs/eval/ft08_fixed.json`, FT07 README, last section):

| | pAUC | caught top 1% / 0.1% | against FT04 | against FT07 |
|---|---|---|---|---|
| FT08, leaky (2026-10-21) | 0.164 | 46.4% / 16.1% | -0.0058 [-0.0106, -0.0010] | - |
| FT08, fixed | 0.1638 | 46.1% / 15.4% | -0.0057 [-0.0101, -0.0013] | -0.0010 [-0.0047, +0.0028] |

Read the last two columns. Against FT04 the fixed FT08 is still a resolved loss: the interval is wholly
below zero. Against FT07 (the same gated loss without blocks) it is a tie: the interval contains zero. So
blocks added nothing on top of gating, and gating had already cost a little. **The verdict "block masking
does not help" stands, now on a clean measurement.**

Why did half the pixels being "seen" change so little? A likely reason, not tested for FT08: part 66
measured that FT04 reproduces pixels it can *see* barely better than ones it must *predict* (set2
background 0.110 visible against 0.137 hidden). A model that cannot copy what it sees gains little from
seeing it, so a leak that shows it the answer barely moves its error.

> **The lesson.** A configuration value read in two places was overridden in one. Nothing crashed, the
> numbers looked plausible, and the experiment's conclusion happened to survive - but nobody could have
> known that without re-running it. When a model needs a non-default inference setting, put it where
> **every** reader of the setting sees it (here: the configuration itself), and write the test that
> fails if any pixel is predicted from a pass that saw it.

## Common confusions

**"The thick rule was chosen to flatter Heavy."** Part 69 covers it: it followed a mask diagnosis (75% of
set4 called thin cloud), it changes FT04 by -0.0004 [-0.0015, +0.0015], and it was adopted after seeing
results - one more reason a release wants more than four scenes.

**"Parity proves Heavy is good."** Parity proves the *product path* gives the *evaluator's* numbers. Whether
those numbers are good is the bench's question (part 69).

**"16 threads, so 8 times faster."** 4 times, measured. Some work does not split and threads share memory.

**"The Dockerfile sets 2, so the worker runs on 2."** Inside compose it runs on 16: compose's
`environment:` replaces the image's `ENV`, and `setdefault` never overwrites a value already set. Outside
compose it runs on 2.

**"The leak inflated FT08's score, so FT08 was really worse."** The fixed run is 0.1638 against the leaky
0.164, a change smaller than the bench resolves. The leak was real; its effect on this model was tiny.

**"The fix changes FT04 or Heavy."** Neither has `mask_block` above 1 (their configs default to 1), so the
`if` is false and nothing changes for them.

**"`checkerboard_cell_size` is in pixels."** It is in tokens of 4 x 4 pixels (part 65's source check). A cell
of 2 is 8 x 8 pixels.

## Check yourself

1. A pixel's head probabilities are water 0.3, thick cloud 0.6, thin cloud 0.4, shadow 0.8, snow 0.0. What
   class does it get under the old rule, and under the thick rule? Is its score zeroed in either case?
2. Recompute the mean of the evaluator's four parity pAUCs, showing each addition.
3. Using Amdahl's law and `p = 0.923`, predict the speed-up from 2 to 24 threads. Why should you not trust
   the answer much?
4. On a 6 x 6-token tile with `mask_block` 2 and the old pass selection (cell 1), how many tokens were
   predicted from a pass that could see them? Show the count per block.
5. In the fixed FT08 row, which interval says "loss" and which says "tie"? What would the FT07 interval have
   needed to look like for blocks to count as a gain?
6. A colleague runs the worker image with plain `docker run` and finds Heavy takes 30 minutes a scene. What
   happened, and which variable fixes it?

<details>
<summary>Answers</summary>

1. Water 0.3: no. Shadow 0.8 > 0.5: class 3. Snow: no. Old rule `max(0.6, 0.4) = 0.6 > 0.5`: class 2,
   cloud, zeroed. Thick rule `0.6 > 0.5`: also class 2, cloud, zeroed. Both rules agree whenever thick
   cloud alone passes 0.5; they differ only for pixels that are thin cloud and not thick.
2. `0.26165 + 0.18742 = 0.44907`; `0.44907 + 0.08609 = 0.53516`; `0.53516 + 0.22167 = 0.75683`;
   `0.75683 / 4 = 0.18921`, 0.1892.
3. 2 threads: `(1 - 0.923) + 0.923 / 2 = 0.077 + 0.4615 = 0.5385`. 24 threads:
   `0.077 + 0.923 / 24 = 0.077 + 0.0385 = 0.1155`. Speed-up `0.5385 / 0.1155 = 4.66` (4.67 with `p` unrounded). Not to be trusted:
   `p` was fitted to one measurement, the law ignores memory bandwidth, and 24 threads on a 24-core host
   leaves nothing for the api and the database - the compose comment chose 16 for that headroom.
4. A 6 x 6-token tile has 3 x 3 = 9 blocks of 2 x 2 tokens. In each block the cell-1 pattern takes 2 tokens
   from the right pass and 2 from the wrong one: `9 x 2 = 18` of 36 tokens, half.
5. Against FT04, -0.0057 [-0.0101, -0.0013]: wholly below zero, a loss. Against FT07, -0.0010
   [-0.0047, +0.0028]: contains zero, a tie. For a gain the FT08 - FT07 interval would have had to lie wholly
   above zero.
6. Outside compose nothing sets `OMP_NUM_THREADS` and friends, so the image's `ENV` (2) applies. Pass
   `-e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 -e NUMEXPR_NUM_THREADS=16`
   (`WORKER_THREADS` is read only by the compose file).

</details>

---

Next: part 71, the spectral namer - a second model built on V2's stem.
