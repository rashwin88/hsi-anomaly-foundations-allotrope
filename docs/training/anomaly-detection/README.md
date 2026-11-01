# Finding anomalies in satellite imagery

A course for someone who has never worked with hyperspectral data and has never opened this
repository. Start at part 1 knowing nothing; finish able to read the source.

Thirty-four parts, each finishable in a sitting. **Read them in order.** Part N assumes parts
1 to N-1 and nothing else, so skipping ahead will cost you more time than it saves.

| part | one-sentence purpose |
|---|---|
| [01](01-the-problem.md) | What problem this system solves, and why it is hard |
| [02](02-light-bands-cubes.md) | What a hyperspectral cube actually is |
| [03](03-scene-project-action.md) | Where you are: the product's four nouns |
| [04](04-the-sensors.md) | Six sensors, and what each one hands you |
| [05](05-the-vendable.md) | The one data structure everything downstream consumes |
| [06](06-anomaly-as-surprise.md) | The core idea, and why there are no labels |
| [07](07-the-band-pipeline.md) | Making five sensors comparable |
| [08](08-rx-detection.md) | Statistical detection, and one banned technique |
| [09](09-local-rx.md) | Scoring a pixel against its neighbours |
| [10](10-reconstruction-models.md) | Teaching a network what normal looks like |
| [11](11-masking.md) | Why a pixel must never help reconstruct itself |
| [12](12-residual-to-score.md) | Turning reconstruction error into a decision |
| [13](13-end-to-end.md) | One scene, one anomaly, start to finish |
| [14](14-material-and-export.md) | Naming the material and getting it out |
| [15](15-naming-the-family.md) | Naming the material family of a pixel the object only partly fills |
| [16](16-detection-measures.md) | Measuring detection where the analyst looks: pAUC, TPR at 0.1%, objects caught, and chance |
| [17](17-public-benchmarks.md) | The public benchmarks (HAD100, ABU), and why none of them tests this product |
| [18](18-the-implant-benchmark.md) | The implanted-target benchmark: real held-out scenes, synthetic objects like the real ones |
| [19](19-running-an-evaluation.md) | Running an evaluation: one config, scene-major loop, cached score maps, chance - and a classical baseline that wins on the implants and loses on real objects |
| [20](20-bench-v2.md) | bench-v2: copying contrast, foreignness and halo from the real objects, solving on the realised object, and the ordering test that lets a benchmark be trusted |
| [21](21-scope-and-masks.md) | Scope: cloud excluded, water kept for vessels, and what a mask costs - the 39 real objects inside FT03's mask, and set4's "desert" that was haze |
| [22](22-the-sorting-bug.md) | How a sorting bug rewrote the scoreboard: `np.interp` needs ascending wavelengths, PRISMA lists them descending, and the cross-check against the product path that found it |
| [23](23-using-indradhanu-v2.md) | Using Indradhanu V2 in an action chain: a native-band band filter (the default since 2026-10-31), the codename, the error you get on a 165-band filter, the new cloud mask and class map, and how one action now runs both Indradhanus |
| [24](24-scoring-gains.md) | Three ways to improve the score on a fixed model - a spatial matched filter, rank fusion with PCA-RX, a mixture of backgrounds - and why each lost real objects: pixel measures against object measures, rank max's false alarms, and bench-v2.1's placement bias |
| [25](25-gated-scene-adaptation.md) | Gated per-scene adaptation: GT-HAD's gating and early stopping transplanted onto FT04, choosing settings without ground truth (lr from the loss, N by leave-one-scene-out), and why it failed - the gate changed nothing because the targets were never learned, and a pretrained model has no early window |
| [26](26-natural-first-then-the-class.md) | Naming materials, redesigned (2026-10-31; built as part 27): where the ten man-made families came from (regex on file names - by use, not material; glass has one spectrum; pavement and building 0.51 degrees apart), what can be named at 30 m, natural first then the class with P(class) = P(man-made) x P(class \| man-made), level 1 as "not explained by nature" so camouflage is flagged, ACE on three bands and why fill drops out, a matching model instead of a classifier so a new library needs no retrain, and the separability check |
| [27](27-the-namer-as-built.md) | The spectral namer as built (2026-10-31; N001 stopped at epoch 1, N001b training): a 0.35 M-parameter (347,685) dual encoder on Indradhanu V2's stem and post-stem BatchNorm, the 3-7 step ring mean (15 x 15 minus 5 x 5), the planted-sample generator x = g (a t + (1 - a) b) + n with log-uniform fill, smooth gain and neighbour-difference noise, class scores as logsumexp of cosine / tau with the planted spectrum left out, level 1 by matching (natural spectra as one more class, P(man-made) = sigmoid(logsumexp of the man-made class scores - the natural score + l1_bias), because N001's linear head sat at chance; worked on a six-entry library), the fill L1 on every sample, the 20% hash split and the textile/fabric open-set run (N002), and the pass test - man-made recall at ACE's natural false-alarm rate |
| [28](28-producing-a-submission.md) | Producing the AIGC Stage-III submission: boolean 0/1 GeoTIFF + point shapefile and no heatmap, only the prep bundle (`hyper_` / `thermal_`, no score attribute), the PRISMA grid from the HE5's UTM corner attributes (pixel centres: pitch = span / (n - 1), origin minus half a pixel; Set-1 by hand), native bands by default with on-the-fly resampling for the 165-grid model, three results per scene, the 8-connected per-object table (SAM / family / namer), and the reflectance writer |
| [29](29-what-hypersl-is.md) | Why not HyperSL, 1 of 6 - what HyperSL is, read from its code and checkpoint: a per-pixel masked autoencoder (9,027,841 trainable parameters, predicted by hand; 9,158,021 with buffers), a tokenizer of 1-D convolutions over band index, a sin/cos code of (lambda - 400) / 10 with base 100 (the paper says 10,000), mask 0.95 = 7 visible bands of 149, a sigmoid output, pretraining on EnMAP + DESIS L2A min-max scaled per band per scene; our reproduction (Chikusei K-means OA 59.6 vs 60.2, ARI 0.5404 vs 0.5405) and what it is genuinely good at |
| [30](30-min-max-and-the-library.md) | Why not HyperSL, 2 of 6 - per-scene min-max on a three-pixel scene by hand (same spectrum, two inputs; one bright pixel moves everybody; a library spectrum has no frame and borrowing one is circular; spectral angle 15.07 -> 36.87 deg), a convolution over band index spanning 217 nm across PRISMA's 1.4 um gap, and the 2026-09-23 matching run: below SAM on 3 of 4 scenes, one sage-leaf spectrum the best match for 59.6% of set2's anomalies |
| [31](31-hypersl-anomaly-detection.md) | Why not HyperSL, 3 of 6 - anomaly detection: spectral self-completion on a 3-band toy (a smooth anomaly rebuilds itself, a neighbour cannot rebuild it), HyperSL's masked-reconstruction residual measured on the 909-object PRISMA bench with the evaluator's own measures beside Indradhanu FTH3 epoch 5, PCA-RX and MNF-RX (2026-10-31, `research/experiments/hypersl-anomaly-bench/`), and HyperSL embeddings as a second-stage false-alarm filter |
| [32](32-hypersl-unmixing.md) | Why not HyperSL, 4 of 6 - "endmember-free" unmixing on HyperSL: VCA + FCLS wins every exact-truth set, the authors' code starts from VCA, a random backbone equals the pretrained one, R_theta carries 29-56% of the reconstruction; HySU object AUC 0.68 / 0.77 at 0.25 / 0.5 m against ACE's 0.97 / 1.00, explained by hand (3x3 averaging gives a / 9; a TV penalty subtracts 2 lambda / (k c), so small targets go to zero); 5-13 min per scene on 96 GB |
| [33](33-hypersl-cloud-mask.md) | Why not HyperSL, 5 of 6 - a cloud classifier on HyperSL embeddings + 7 raw bands finds 28.9% of EnMAP-labelled cirrus against 67.9% for the 148 raw bands (both confusion tables in full, recall and precision by hand), why: pretraining on near-cloud-free scenes, and min-max turning one cirrus layer into a step of delta / span that differs threefold between scenes |
| [34](34-hypersl-verdict.md) | Why not HyperSL, 6 of 6 - the verdict: a decision table of requirement, HyperSL, our models and the evidence; the costs (GPU time, licence, data on Baidu only); what would have to change for HyperSL to be useful here, and what we kept from it |

## How to use this

**Do the arithmetic.** Every part ends with check-yourself questions, and at least one in
each needs a pen. The worked examples are deliberately tiny - three bands, a 2x2 grid - so
you can do them in a minute. Reading them is not the same as doing them.

**Open the files.** Every claim points at a path you can open. Where a source comment and
the code disagree, this course says so and follows the code - those disagreements are worth
more to you than the agreements.

## What this is not

Study material, subordinate to two things:

- **The source.** If this course contradicts the code, the code is right.
- **`docs/01-orientation.md` through `docs/10-code-style.md`**, which are the reference
  documentation, plus `docs/lld/` for individual subsystems.

This is a path *into* those, not a replacement.

## Values that will drift

Some numbers here are read from configs and checkpoints and will go stale. Re-verify before
relying on them:

| part | what will drift |
|---|---|
| 04 | band counts per sensor |
| 07 | `BandFilterConfig` defaults, the 165-band grid |
| 08, 09 | detector defaults - windows, regularisation, component counts |
| 10 | parameter counts, checkpoint validation losses |
| 12 | scoring defaults and threshold percentiles |
| 15 | family counts in the library, `family_alpha`, the ~2.5 ms per pixel cost, PRISMA results |
| 16 | `compute_roc`'s threshold count formula, the FT02b bench measures |
| 17 | the state of HAD100 / ABU (ABU's page was already 404 on 2026-10-16), the leakage distances |
| 18 | bench-v1's object counts and every results row - a rebuild (new hosts, new seed) or a new model moves them; the tier edges 0.502 / 0.934. Since 2026-10-18 its v0.2.0 row is void and its signature implants are known garbled (part 22) |
| 19 | every stage-3 number (exclusion, PCA-RX), the diagnosis medians (0.006 / 0.74); dated update notes record what changed since (cache key, `dirty` flag, the 39 objects, bench-v2 accepted, the resample bug) |
| 20 | every bench-v2 number (2,913 objects, the model and PCA-RX rows), the `real_v2.json` medians (0.0071 / 0.643), the per-tier ratios recomputed from `objects.json`, sigma 1.4, the 0.25 -> 0.02 s subspace timing. The v0.2.0 reading question was resolved on 2026-10-18 (part 22): its v0.2.0 row is void and bench-v2 is superseded by bench-v2.1 |
| 21 | the cloud-only and water + cloud numbers, the per-scene object counts, the 39-object breakdown and FT03's 11% set4 "cloud" (haze, corrected 2026-10-18); all move with a new segmentation head - FT04 passed on 2026-10-18 and its masks are the release candidate's |
| 22 | every bench-v2.1 number (2,817 objects, the model rows, the 1.05 / 1.15 / 1.60x foreignness), the resample-check rows, the PRISMA band orders quoted from the dev laptop's files, NumPy 2.5.2's `np.interp` behaviour on unsorted input (undefined by contract - another version may differ) |
| 23 | Indradhanu V2's status (the band-filter default became native bands on 2026-10-31 and the "one action cannot run both" limit went with it; the scoring codename default is still v0.2.0), the 0.165 / 0.170 like-for-like, set4's 11.83% excluded, the missing viewer panel; the CPU cost, unmeasured when written and since measured (454 s per PRISMA scene for Heavy on 16 threads, 2026-10-29, update note at the top of the part) |
| 24 | every scoring-gains number (FT04's 0.162 baseline, the nine scorer rows, the per-size objects caught, the 60-70% / 5% / 5.2% target-clustering figures, the 13 cache files); the verdict "RM stays" holds until a scorer or model passes the object-level criteria. The uncertainty head was run as FT06 on 2026-10-20 and failed (Indradhanu part 63; research/experiments/indradhanu-finetune/FT06-uncertainty/) |
| 25 | every scene-adaptation number (the 0.170 bar, the four variant rows, the per-scene pAUCs and the 0.1565 leave-one-scene-out headline derived from them, the 4.55% gate coverage, the lr-probe losses, 1.6 steps/s), the GT-HAD reproduction figures (5.5% / 2.0% PD at 1e-3, 0.873 / 0.843 mean valid AUC); the verdict holds for FT04 and this adaptation recipe only - a from-scratch or differently pretrained model would need its own run |
| 26 | the design (2026-10-31, now `docs/proposals/spectral-namer.md`; the built model departs from it, part 27), the class list, the family counts from `data/family_library/index.json` (411 spectra, `FAMILY_CAP = 40`), the family angles (from the step-by-step book's compute script), and every separability number (PRISMA 19,728 and EnMAP / AVIRIS-NG 14,796 plantings per fill, all final on 2026-10-31; the README's "~12%" for pavement is 10.4% in the JSON) |
| 27 | everything about the namer: N001 / N001b / N002 settings (`configs/namer/`; `"level1": "matching"` in N001b and N002, default `"head"`), the bank counts (378 / 159 / 34 on EnMAP after the 2026-10-31 de-duplication; 408 / 174 / 41 before), the per-class held-out shares (17-30%), the 347,685 parameters (347,684 before `l1_bias`), the CPU smoke numbers, N001's epoch-1 level-1 figures and the laptop head-vs-matching comparison (N001 README), N001b's epoch-1 figures and ACE re-scored on the v1 classes (N001b README), the pass bars (ACE + 0.10 from `separability_<sensor>.json`) - and the final results, pending when written |
| 28 | the submission status (C1-C7 on the local stack only, deploy not approved), the deadlines (report 2 Nov, presentation 4 Nov 2026), the Set-1 grid offset (median 2.4 px, 95% 4.1, max 5.6), the three suggested results, the pydantic default `True` and its description (the `_check_band_grid` and resolver comments were corrected on 2026-10-31) |
| 29 | the HyperSL release (checkpoint `10_base_mask95`, epoch 10, upstream commit `e56de7e`), the licence status (no LICENSE on GitHub, MIT on Hugging Face), the data location (Baidu Netdisk), and every reproduction number (validation of 2026-10-02/03, `2d893e6`) |
| 30 | the 2026-09-23 matching run's numbers (ECOSTRESS 1,947 / USGS 518 entries, outputs outside the repo in `hypersl_match_data/results/`), the 28% band-sampling figure (no output file kept), the kernel-span table (band lists of 2026-10-31) |
| 31 | the HyperSL bench run of 2026-10-31 (149-band axis, mask 0.95, 4 draws, seeds 1000-1003, RTX 4000 SFF Ada), the FTH3 epoch 5 comparison numbers (eval of 2026-10-29), the false-alarm-filter numbers (2026-09-26, outputs outside the repo) |
| 32 | every unmixing number (Colab run of 2026-10-13/14, README in `4d292ac`, raw outputs on the team Drive), the assumed lambda and c of the TV example |
| 33 | both confusion tables and the class shares (2026-10-01, `b452e8d`; training tables on the GPU box, not in the repo), the assumed cirrus delta and spans |
| 34 | the decision table's numbers (all of 29-33), the runtimes, the licence and data-access facts |

Structural claims - what a vendable is, why masking works the way it does - will outlive
the numbers.

## A note on where this lives

The house style says training modules go in a gitignored working folder. This one is tracked
instead, because it exists to onboard people and a doc that does not survive `git clone`
cannot do that. It is separated from `docs/01-10` so nobody mistakes a course for reference
documentation.
