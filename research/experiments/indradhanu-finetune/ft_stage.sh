#!/bin/bash
# Stage the fine-tuning data on the kept L40S (prepared 2026-10-16; run only on the user's go):
# waits for the F001 driver to finish, frees the SpectralEarth cache, then pulls the three shard
# sets (EnMAP + PRISMA re-encoded to i16dz, AVIRIS-NG native) and the band lists into /opt/dlami/nvme/ft.
# Same region, no egress. ~233 GB. DONE holds ok|failed and a timestamp.
set -uo pipefail
N=/opt/dlami/nvme; P=s3://allotrope-training-data-prod-india/patches; L=/var/log/ft-stage.log; rm -f $N/ft/STAGED
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
until grep -q "F001 driver finished" /var/log/f001.log 2>/dev/null; do sleep 120; done
pgrep -f train_foundation_model >/dev/null && { log "a trainer is still running - stop"; exit 1; }
log "F001 done; freeing the SpectralEarth cache"; rm -rf $N/se_cache $N/se_stage $N/se_stage.BLOCK
mkdir -p $N/ft/band_lists
aws s3 cp --recursive s3://allotrope-training-data-prod-india/experiments/indradhanu-finetune/band_lists/ $N/ft/band_lists/ --only-show-errors
for spec in "enmap enmap/native-v1z" "prisma prisma/native-v1z" "aviris_ng aviris_ng/native-v1"; do
  set -- $spec
  for split in train test; do
    log "pulling $1/$split"
    aws s3 cp --recursive $P/$2/$split/ $N/ft/$1/$split/ --exclude "*" --include "*.tar" --only-show-errors \
      || { log "pull of $1/$split failed"; echo "failed $(date -u +%FT%TZ)" > $N/ft/STAGED; exit 1; }
  done
done
for d in $N/ft/*/train $N/ft/*/test; do log "$(ls $d/*.tar | wc -l) shards in $d"; done
df -h $N | tail -1 | tee -a $L
echo "ok $(date -u +%FT%TZ)" > $N/ft/STAGED
