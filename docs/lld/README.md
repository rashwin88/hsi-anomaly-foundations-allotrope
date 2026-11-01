# Low-level designs

One file per subsystem whose *why* is not recoverable by reading it.

These are not API docs. The numbered docs (`docs/01-orientation.md` onward) tell you what
the system does; an LLD here tells you why one piece of it is built the way it is, what
must stay true, and what breaks if somebody tidies it up.

A subsystem earns a file here when it was newly added, or when its rationale lives only in
someone's head. Small changes to existing behaviour update the existing file rather than
adding another.

| file | subsystem |
|---|---|
| `action-api.md` | the five modules `actions.py` was split into |
| `pixel-stats.md` | resolving per-band normalisation statistics |
| `local-background.md` | batched Mahalanobis for the annulus detectors |
| `frequency-destriper.md` | the FFT notch filter and its significance test |
| `test-harness.md` | how anything gets verified in this repo |
| `segmentation-sharding.md` | the `enmap_seg` lane that attaches provider labels to shards |
| `inventory.md` | the raw-dataset inventory: checksum chain, manifest-as-truth, Supabase lockout |
| `spectralearth.md` | SpectralEarth patches: the derived band table, split by site, native vs common-grid loading, shards |
| `spectral-stem.md` | the swappable spectral stem (linear / gate / cross-attention / reduced attention), loader hooks, band alignment |
| `enmap-shards.md` | EnMAP L2A at native bands with quality labels, from the inventory, for fine-tuning and a cloud/water head |
| `spectral-family-match.md` | naming the material family of 30 m anomaly pixels: family library, Simplex ACE, calibration and the alpha / p\* bar, evaluation, integration recipe |
| `spectral-namer-in-product.md` | the spectral namer in `spectral_library_match`: objects, natural vs man-made before matching, closest known class + tentative flag, files, export `/3`, viewer |
| `evaluation.md` | `app/evaluation` + `scripts/evaluate.py`: models x datasets x scorers x measures from one config, score-map cache, the stage-3 reference run |
| `implant-bench.md` | the implanted-target benchmark: contrast tiers, the strength solve, bench-v1, bench-v2, the 2026-10-18 descending-wavelength corrections and bench-v2.1 |
| `finetune.md` | the fine-tuning trainer and segmentation head: stages, mixed batches, pseudo-labels, class weights, implants, loader workers |
| `generic-hsi.md` | the `generic_hsi` sensor: any GDAL hyperspectral file, band-table discovery, FWHM / scale / no-data decisions, `.igm` georef, the pre-flight report |
| `segmentation-v2.md` | `scene_segmentation` method `indradhanu_v2`: V2's head as its own step, class_map / keep_mask / class_codes.bin, scoring that reuses it instead of running the head again, the viewer |
| `story-gif.md` | the story GIF of a match action: frames from the chain's files, the endpoint, its cache, why it is never in an export |

## How these are written

Every claim points at a file you can open, and was checked against the source rather than
against an existing doc. Where a source comment and the code disagree, **the code wins and
the disagreement is written down** - those contradictions are the most useful thing here,
because they are what will mislead you at 2am.

Numbers are concrete. "Uses a lot of memory" is not a statement; "24 bytes per padded pixel,
52 MB per padded PRISMA band, 12.4 GB for all 239 at once" is.

Values that drift are marked where they appear. Config defaults, checkpoint metrics and
line counts all rot; if one matters to your decision, re-verify it.

These are subordinate to the source and to `docs/01-09`. If they contradict either, they
are wrong.
