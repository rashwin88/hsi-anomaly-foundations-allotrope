# Running the SpectralEarth pretraining experiment on the airgapped H200

**Status:** detailed plan, 2026-10-14. Nothing built yet. Agree it before any chunk starts.
**Operator:** Ashwin, end to end - builds the disk, carries it, runs on the VM, types results back.
**Window:** the H200 for one month from 2026-10-14, 24x7.
**Code:** our branch only (`feat/indradhanu-se-pretrain`). The 2026-10 H200 trial branch is not
used; only the facts its trial established are (section 0). Machine facts:
`research/experiments/indradhanu-h200/H200_SYSTEM.md` on that branch - read, not merged.

The constraint that shapes everything: **one disk in, nothing out.** A missing file, a wrong
path or an untested command costs a disk round trip. So the exact disk is rehearsed here
first - laptop GPU, Docker, `--network none`, the VM's uid - before any data is copied.

---

## 0. What the October trial already taught us (facts, not code)

| Fact | Consequence for us |
|---|---|
| Ubuntu 22.04 VM, 16 vCPU, 64 GB RAM, 1x H200 NVL 143 GB, driver 580 / CUDA 13.0 | image may use any CUDA <= 13.0; torch 2.5.1+cu124 is what we tested |
| airgapped: no S3, pip, apt, `docker pull`, W&B | everything on the disk; trainer must never try S3 (section 6, A1) |
| disk read ~520-540 MB/s O_DIRECT; write ~430 MB/s | ~80 patches/s at 6.6 MB; the data rate, not the GPU, sets epoch time for A and B |
| VM user is **uid 1001**, image users default to 1000 | torch's inductor cache calls `getpwuid` -> crash; run with `-e USER -e LOGNAME -e HOME=/tmp` |
| user not in the `docker` group | `sudo usermod -aG docker $USER` once, or `sudo` every command |
| default `/dev/shm` 64 MB -> "bus error" | `--shm-size=32g`; workers x 2 prefetch x batch x 6.6 MB must stay well under it |
| batch 256 x 12 workers exhausted shm in the trial | our batch is 4; 6 workers is plenty |
| EnMAP files mix `-METADATA.XML` / `.xml` | our `find_member` must be case-insensitive (A3) |
| `keep_top_k` by a loss that changes scale deleted the final checkpoint in the trial | ours keeps `last.pt` and `best.pt` outside top-K cleanup (built 2026-10-14) |
| results come out only as typed text | every command ends with a short block that can be typed in a minute |
| v0.2.0 on the 4 PRISMA benchmarks: mean combined AUC **0.674** | reference number our eval must reproduce for v0.2.0 before we trust it on A/B/C |

## 1. Data transfer strategy

**Measured 2026-10-14:** S3 -> this laptop **98 MiB/s** (~3 h per TB). S3 egress to the internet
from ap-south-1 **$0.109/GB** (first 10 TB). VM: **2 TB disk, 64 GB RAM**, disk read ~540 MB/s.

**Principles**
1. **Compress before anything leaves AWS.** Re-encode the shards in-region on the prod box (S3 ->
   EC2 in the same region is free) as **band-delta + zstd-3**: each pixel stored as its difference
   from the previous band, then zstd. Measured on real shards: **1.91x smaller**, exact round trip
   (modular int16 arithmetic), decode ~140 MiB/s per core. This halves the egress bill *and*
   nearly doubles what the VM disk delivers: ~540 MB/s compressed = ~155 patches/s instead of ~80.
   The extra S3 copy costs ~$13/month and is deleted after the run.
2. **Pull each byte out of AWS once.** One download onto the external disk; that disk is also the
   archive for later trips and reruns. Nothing is downloaded twice.
3. **Take only what the plan uses.** 1,000 of 3,419 SpectralEarth train shards (150k patches),
   20 test shards. More only if pretraining shows it is data-starved.
4. **Data that is not in S3 stays out of S3.** PRISMA for fine-tuning comes from the Google Drive
   copies (510 `.he5`, already outside AWS) and is sharded on this laptop - $0 egress - not from
   the Beta bucket (223 scenes, ~470 GB, ~$51). The image, benchmarks and v0.2.0 checkpoint are
   built or held locally.
5. **Verify at every hop** against `manifest.json` sha256s (of the compressed shards).

**Budget**

| Item | Raw | Shipped (compressed) | Egress |
|---|---|---|---|
| SpectralEarth train, 1,000 shards | 0.97 TB | ~0.51 TB | ~$56 |
| SpectralEarth test, 20 shards | 19 GB | ~10 GB | ~$1 |
| EnMAP shards (210 scenes, ~14k patches) | ~90 GB | ~47 GB | ~$5 |
| PRISMA shards (from Drive, trip 2) | ~150 GB | ~80 GB | $0 |
| Image, configs, benchmarks, baseline | ~15 GB | ~15 GB | $0 |
| **Total on the VM** | | **~0.66 TB of 2 TB** | **~$62** |

Uncompressed, the same data would cost ~$122 and run at half the patches/s.

**Trips**
- **Trip 1 (~day 5):** image, configs, benchmarks, baseline, SpectralEarth 1,000 + 20 shards,
  EnMAP shards. Enough for the comparison (E001-E003), full pretraining, and EnMAP fine-tuning.
- **Trip 2 (~day 10-12):** PRISMA shards, a new image if code changed (fine-tuning, eval
  fixes, the UI if ready). Reuses the same disk.

## 2. Pre-training strategy

**P0 - comparison (E001 linear, E002 gate, E003 cross-attention).** 200 train shards (30k
patches), 20 test shards, 30k samples per epoch, B0 SegFormer, D = 32, lr 5e-4 cosine, bf16, early
stop 1% / 10 epochs / minimum 20 / ceiling 200. Then the 4-scene PRISMA eval and the decision rule.
- **Batch on the H200:** 32 per step, no accumulation, for all three (C needs ~3.7 GiB per patch:
  32 x 3.7 = ~118 GB of 143). Same effective batch as the AWS config (4 x 8) but BatchNorm sees 32
  patches instead of 4 - better. **All three runs must use the same batch setting** - if E003
  ends up on an AWS L40S (48 GB, batch 4 x 8), E001/E002 on the VM use 4 x 8 too.
- **Time:** A and B ~155 patches/s (disk-bound) -> ~3-4 min per epoch; C ~75/s (compute-bound) ->
  ~7 min. 20-80 epochs each -> **E001+E002 ~1 day, E003 ~1 day.**

**P1 - full pretraining of the winner.** 1,000 shards (150k patches), epoch = one full pass, ~60
epochs = ~9M samples (v0.2.0 saw ~8M), cosine over 60, same early-stop rule, and **the PRISMA
eval every 10 epochs** - reconstruction loss is not the goal; a model that reconstructs too
well rebuilds anomalies too. Keep two checkpoints: best validation loss and best mean ROC-AUC.
- **Time:** C ~33 h, A/B ~16 h.
- **Model size:** B0 as in P0. A B1 (15.5M) ablation only if P1 finishes early - the GPU has
  headroom, the comparison does not need it.

## 3. Fine-tuning strategy

**Batching (decided 2026-10-14):** pre-training uses single-sensor batches only (SpectralEarth,
202 bands). Fine-tuning may mix sensors in a batch; the trainer can align every sample onto one
band list by wavelength (`band_list_path`, built 2026-10-14), and what remains open is the
label-key policy for mixed batches (EnMAP samples carry `label_*`, SpectralEarth/PRISMA do not) -
decided with the segmentation head.

**F1 - EnMAP + PRISMA** from P1's best-AUC checkpoint.
- **Data:** EnMAP shards (trip 1), PRISMA shards (trip 2), plus **20% SpectralEarth replay** in
  every batch so the model does not forget the global distribution. The 4 benchmark scenes are
  excluded by scene id (checked at sharding).
- **Clouds:** EnMAP pixels labelled cloud or cirrus (outside the >4,000 m low-water-vapour
  scenes where those labels are false) are dropped from the loss; a trainer option reading the
  `label_*` keys. PRISMA L2D has no masks - no screening on trip 2; revisit with the segmenter.
- **Bands:** if C wins, each sensor is read at its own wavelengths (PRISMA ~230 native bands,
  EnMAP 202) - no resampling. If A or B wins, PRISMA is resampled to SpectralEarth's 202.
- **Optimisation:** lr 1e-4 (1/5 of pretraining), cosine over 30 epochs, same early-stop rule,
  held-out validation by datatake (EnMAP) and by scene (PRISMA), PRISMA eval every 5 epochs.
- **Done when:** the fine-tuned model beats v0.2.0's mean combined AUC 0.674 on the 4 scenes, or
  the plateau shows it cannot.

**F2 - cloud/water head (later).** The EnMAP shards already carry the labels and scene facts.

## 4. Training from the product UI (separate workstream, after trip 1)

Requested 2026-10-14: launch the app on the VM and train by clicking, with models loaded from a
local folder, not only from the `allotrope_models` volume. This is a product feature, so it gets
its own design doc and a `final design/` spec entry before code. First questions:
- The full stack on the VM (postgres, api, worker, frontend) shipped as saved images; GPU in the
  worker container.
- A Training page: pick a local shard set + config, start/stop/resume, live metrics (polling
  `metrics.csv`, matching the frontend's polling style), run the PRISMA eval.
- The trainer as a worker job on the Postgres queue, so it survives api restarts.
- "Models from local": register a checkpoint from a folder on the VM into the model catalog
  (today the api reads `<arch>/current.json` manifests from the `allotrope_models` volume).
- Not on trip 1: the CLI kit is the proven path, and a UI bug would cost a trip.

## 5. Timeline and critical path

| Day | Code (here) | Data (here) | VM |
|---|---|---|---|
| 1 | A6 compressed shards, A1 run control | B0 re-encode on the prod box | - |
| 2 | A2 eval (reproduce 0.674), A3 EnMAP sharder | B1 download (~1.5 h), B2 verify | - |
| 3 | A4 image + Trivy, A5 kit | B3 EnMAP shards built on prod box, downloaded | - |
| 4 | **C rehearsal** (section 9); fix what it finds | B4 final copy onto the external disk + verify | - |
| 5 | - | - | install, verify-data, selftest, E003 starts |
| 6-7 | update run folders from typed results | - | E003 -> E001 -> E002, evals |
| 8+ | decide the winner (rule in the design doc) | - | full pretraining of the winner (~1-2 days), then EnMAP fine-tune |

Critical path: A2 (eval) and C (rehearsal). The download (B1) runs in parallel from day 1.

## 6. Workstream A - code (on our branch, nano-chunks)

### A1. Run control for an offline, typed-back run
- **Offline guard.** `hot_storage.offline: true` in our trainer: never call `_ensure_local_shards`
  (it falls back to `aws s3 cp` when counts differ - a confusing crash without network); instead
  check the local shard counts against the config and fail with one clear line.
- **W&B off** by default in the VM configs (`wandb.enabled: false`).
- **`summary`** - one block per run from `metrics.csv` + `last.pt`: stopped/finished, epochs,
  best epoch, best val L1, best val SAM (deg), last-5-epoch trend, patches/s, peak GPU GB, wall
  time, and the sha256 prefix of `best.pt` (proves which checkpoint was evaluated).
- **`status`** - is training alive (pid + last log line age), last 5 metrics rows.
- **Acceptance:** unit tests for the summary formatter; offline guard tested with a missing shard.

### A2. Evaluation on the 4 labelled PRISMA scenes (new script, ours)
`scripts/evaluate_se_pretrain.py`:
1. Read each Set's PRISMA L2D `.he5` with `PrismaDatasetBuilder`:
   - for A and B: resampled onto **SpectralEarth's 202 wavelengths**, with `exclusion_ranges`
     overridden - the default (0-450, 912-978, 1131-1152, 1350-1450, 1800-1950 nm) would drop
     SpectralEarth wavelengths A/B cannot run without;
   - for C: the same, **and** a second pass on the 165-band grid (tests "any band list");
   - for the v0.2.0 baseline: the 165-band grid exactly as the product does.
2. Reconstruct the scene in 128 px patches, stride 64, two complementary token masks per patch
   (every pixel predicted while hidden), averaged over overlaps.
3. Residual maps L1 and SAM -> `compute_score` / `compute_roc` from
   `app/utils/anomaly_detection/scoring.py` (reuse, not rewrite).
4. Per scene and model: ROC-AUC (L1, SAM, combined), TPR at 1% FPR, objects caught at the 1% cut,
   with the ground truth dilated by 1 px (it may sit ~1 px off).
5. One table to type back: rows = models (v0.2.0, A, B, C, C-on-grid), columns = 4 scenes + mean.
- **Acceptance:** on the laptop, v0.2.0 reproduces the trial's numbers (mean combined 0.674, set 1
  L1 0.903) within 0.005. If it does not, the eval is wrong and nothing else proceeds.
- Runtime target: < 10 min for all models on the H200.

### A3. EnMAP sharder (chunks 4-8 of the EnMAP plan)
Generalise `write_shard` to any sample; split by datatake stratified on cloud; `scripts/shard_enmap.py`
(pass 1 scene -> tiles on local disk, pass 2 shuffled shards, sidecars, manifest, stats);
case-insensitive file lookup; laptop smoke on 2 scenes; full run on the prod box (CPU).
Ships on this disk if done by day 3; it is not needed for E001-E003.

### A4. Docker image `allotrope/se-pretrain:0.1.0`
- `python:3.12-slim` + pinned `requirements.lock` (torch 2.5.1+cu124, numpy, webdataset,
  pydantic, tqdm, rasterio, h5py, boto3 - boto3 only because shared modules import it).
- Code baked in at `/opt/allotrope` (app/, scripts/) from a tagged commit; the tag is printed by
  `selftest` and by every `summary`.
- `entrypoint.sh` commands: `selftest`, `verify-data`, `train <run>`, `status <run>`,
  `summary <run>`, `eval <run>...`, `shell`.
- Runs as any uid (`HOME=/tmp`, `USER`/`LOGNAME` set), `--network none`.
- **Trivy** scan before it goes on the disk: 0 CRITICAL, 0 HIGH with a fix available, 0 secrets.
- `docker save | gzip` -> ~4-5 GB.

### A6. Compressed shards (section 1)
- `pixels.i16dz` key: band-delta int16 + zstd-3; a webdataset decoder in our trainer turns it back
  into the same `(202, 128, 128)` int16 (exact). Raw `pixels.npy` shards keep working.
- `scripts/recompress_shards.py`: S3 shard -> compressed shard -> S3 (`se-v1z/`), with its own
  `manifest.json`; runs on the prod box (in-region, no egress). ~1,020 shards, ~1-2 h on 4 vCPU.
- **Acceptance:** decoded == original for every sample of a real shard; trainer loss identical on
  raw vs compressed shards; decode keeps up with ~155 patches/s on 6 workers.

### A5. Kit builder `scripts/h200/build_kit.sh`
Assembles the kit folder (section 7), writes `SHA256SUMS` for everything except the shards
(those are checked against `manifest.json` by `verify-data`), and a `KIT_VERSION` file.

## 7. The disk

```
KIT/                                  (external disk root)
  run.sh                              host-side wrapper: docker load / run with the right flags
  RUNBOOK.md                          section 10 of this plan, printed
  KIT_VERSION, SHA256SUMS
  image/se-pretrain-0.1.0.tar.gz      ~4.5 GB
  configs/E001-linear.json, E002-gate.json, E003-cross-attention.json, FULL-<winner>.json
  baseline/hyperspectral_segformer_mae_v0.2.0_epoch200.pt
  benchmarks/set1..4/  scene.he5 + ground_truth.tif               ~8.5 GB
  shards/spectralearth/se-v1/
      train/se-v1/w128_h128_s64/shard-000000..000999.tar          ~0.51 TB (compressed)
      test/se-v1/w128_h128_s64/shard-000000..000019.tar           ~10 GB
      stats.json, manifest.json
  shards/enmap/...                                                ~47 GB compressed (if ready)
```

On the VM everything is copied to `/data/allotrope/` with the same layout; runs write to
`/data/allotrope/runs/<run>/` (metrics.csv, train.log, checkpoints/last.pt, best.pt, eval.txt).

**Space:** ~0.6 TB in on trip 1 (~0.66 TB after trip 2), plus checkpoints (~70 MB each). The VM
has 2 TB; the trial's leftovers (~500-650 GB) may still be there - check `df -h /data`.

## 8. Workstream B - data logistics

| Step | What | Where | Time (est.) |
|---|---|---|---|
| B0 | re-encode 1,020 SpectralEarth shards (A6) | prod box, in-region | ~1-2 h |
| B1 | download the compressed shards + stats + manifest | S3 -> external disk on this laptop (measured 98 MiB/s) | ~1.5 h for ~0.52 TB |
| B2 | `verify-data` against `manifest.json` (sha256 of every shard) | wherever B1 landed | ~1 h for 1 TB on a USB 3 disk |
| B3 | EnMAP shards (A3) | prod box -> S3 -> same disk | ~2 h |
| B4 | copy kit files (image, configs, benchmarks, baseline, run.sh) onto the disk; `SHA256SUMS` | laptop | ~15 min |

Recommended: **download directly onto the external disk** attached to this laptop, with `aws s3 cp`
from an include list generated from the compressed set's `manifest.json` (exactly the 1,020
shards), 16 parallel. Total egress for trip 1 ~$62 (section 1). Our own admin profile is fine on this laptop; no extra key needed. The
disk must be >= 1.5 TB and **exFAT or NTFS** (not FAT32: 4 GB file limit; shards are ~1 GB, the
image ~4.5 GB). The VM is Linux: NTFS is readable there; exFAT too on 22.04.

## 9. Workstream C - rehearsal (day 4, before data goes on the disk)

From a copy of the finished kit with **5 train + 2 test shards** substituted, in WSL Docker on the
laptop GPU, as uid 1001, `--network none`, with the image first deleted from the local Docker:

1. `run.sh install` -> loads the image, checks `SHA256SUMS`, free space.
2. `run.sh verify-data` -> 7 shards OK; then corrupt one byte of a copy -> reported as a mismatch.
3. `run.sh selftest` -> GPU, bf16, one step of each stem, prints the code tag.
4. `run.sh train E003-cross-attention` (config with 2 epochs) -> kill during epoch 2 ->
   rerun -> resumes from `last.pt`, lr sequence continuous.
5. `run.sh status E003-cross-attention`; `run.sh summary E003-cross-attention`.
6. `run.sh eval E003-cross-attention` on 1 benchmark scene -> table prints.
7. The same for E001 and E002 (1 epoch each).
8. Delete a shard from the train folder -> `train` fails in one clear line (offline guard).

Every output must fit on screen and be typeable. Only after all 8 pass does the full data go on.

## 10. Runbook on the VM (what Ashwin types)

```
# once
sudo usermod -aG docker $USER && newgrp docker
cd /media/<disk>/KIT && bash run.sh install          # copies to /data/allotrope, loads image
bash run.sh verify-data                              # ~35 min; must say 1020/1020 OK
bash run.sh selftest
# each run, inside tmux
tmux new -s e003
bash run.sh train E003-cross-attention               # Ctrl-b d to detach
bash run.sh status E003-cross-attention              # any time
bash run.sh summary E003-cross-attention             # when finished -> type back
bash run.sh eval E003-cross-attention                # -> type back the table
# then E001-linear, E002-gate the same way
```

After a reboot or crash: rerun the same `train` command; it resumes.

**Typed back per run:** the `summary` block (~10 numbers) and the `eval` table (5 rows x 5
columns). I record them in `E00x/README.md` and apply the decision rule.

## 11. Failure playbook (fixable without a new disk)

| Symptom | Likely cause | Fix on the VM |
|---|---|---|
| `permission denied ... docker.sock` | not in docker group | `sudo usermod -aG docker $USER; newgrp docker` |
| `bus error` / shm in DataLoader | shm too small | run.sh already sets 32 GB; lower `num_workers` in the config |
| `getpwuid(): uid not found` | env not passed | run.sh sets USER/LOGNAME/HOME; check run.sh was not edited |
| CUDA OOM on C | batch too large for... (not expected at 143 GB) | config `batch_size` 4 -> 2, `gradient_accumulation_steps` 8 -> 16 |
| `verify-data` mismatch | copy damaged a shard | recopy that shard from the external disk |
| training slower than ~60 patches/s | disk contention | `vmstat 5`; nothing else should run during training |
| a config needs a change | - | configs are plain JSON outside the image: edit, rerun |
| a code bug | - | **needs a new image** -> second disk trip; this is what the rehearsal prevents |

## 12. Decision points

1. After the rehearsal: go / fix (day 4).
2. After E003: is C's throughput and loss sane? If C is far slower than ~60 patches/s or diverges,
   stop and rethink before spending days on A and B.
3. After E001-E003 + eval: apply the decision rule (`docs/proposals/spectral-attention-bottleneck.md`):
   C proceeds to full pretraining only if it beats A on mean ROC-AUC; B preferred if it matches C.
4. After full pretraining: EnMAP fine-tune (needs the EnMAP shards, A3) and the final eval against
   v0.2.0's 0.674.

## 13. Open questions

- [x] H200 window: one month from 2026-10-14, 24x7.
- [x] Operator: Ashwin, end to end. Trial branch not used.
- [ ] VM free space now, and may the trial's leftovers be deleted? (`df -h /data`, `du -sh /data/*`)
- [ ] External disk: capacity (>= 1.5 TB), filesystem (exFAT/NTFS), USB 3?
- [ ] Download machine: this laptop (recommended) - what is its line speed to AWS Mumbai?
- [ ] Does the VM mount USB disks itself, or does an admin copy them in (the trial's kit was
  "pushed by the admin onto the VM's own disk")? This decides whether `install` copies from a
  mounted disk or from a folder already on `/data`.
