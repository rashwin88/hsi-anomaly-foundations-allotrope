#!/bin/bash
# F001 driver on the kept g6e.2xlarge (decided 2026-10-15): 200 epochs of the final SpectralEarth
# pretraining, the training shards rotated every 20 epochs through blocks of 200 (30,000 patches).
#
#   setsid nohup bash /opt/rotate_f001.sh > /var/log/f001.log 2>&1 < /dev/null &
#
# Why this works without trainer changes: with hot_storage.skip_sync_if_exists the trainer trains on
# whatever *.tar files sit in its local train folder, and auto_resume continues from last.pt with the
# optimizer, scheduler and early-stop state - so a rotation is: stop after epoch 20k (its checkpoint is
# already in S3 when metrics.csv gets the row), swap the folder's contents, resume.
#
# Block k (0-based) = train shards k*200 .. k*200+199; block 0 is already on disk from E001-E005. The
# next block downloads into $STAGE while the current one trains (same-region S3, no egress charge).
# Re-running this script after an instance restart picks up where it stopped: everything it needs is
# derived from metrics.csv (epochs done) and the BLOCK marker files.
set -uo pipefail
PREFETCH=""
N=/opt/dlami/nvme; R=F001-final-pretrain; D=$N/runs/$R
B=allotrope-training-data-prod-india; SHARDS=s3://$B/patches/spectralearth/se-v1/train
OUT=s3://$B/experiments/indradhanu-se-pretrain/$R
TRAIN=$N/se_cache/train/se-v1/w128_h128_s64; MARK=$N/se_cache/train/BLOCK
STAGE=$N/se_stage; STAGE_MARK=$N/se_stage.BLOCK
PER_BLOCK=200; EPOCHS_PER_BLOCK=20; TOTAL=200
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /opt/pytorch/bin/activate
mkdir -p $D
log() { echo "$(date -u +%FT%TZ) $*"; }
epochs_done() { [ -f $D/metrics.csv ] && echo $(($(wc -l < $D/metrics.csv) - 1)) || echo 0; }

fetch_block() {   # fetch_block <k> <dir>: the block's 200 shards, 16 at a time; fails unless all 200 arrive
  local k=$1 dir=$2
  mkdir -p $dir
  seq $((k * PER_BLOCK)) $((k * PER_BLOCK + PER_BLOCK - 1)) | xargs -P 16 -I{} sh -c \
    'f=$(printf "shard-%06d.tar" {}); [ -s '"$dir"'/$f ] || aws s3 cp '"$SHARDS"'/$f '"$dir"'/$f --only-show-errors'
  [ $(ls $dir/*.tar 2>/dev/null | wc -l) -eq $PER_BLOCK ]
}

aws s3 cp $OUT/config.json $D/config.json --only-show-errors || { log "no config in S3"; exit 1; }
[ -f $MARK ] || echo 0 > $MARK          # what E001-E005 left on disk is block 0 (shards 0-199)
( while true; do sleep 300            # mirror the record to S3 every 5 minutes
    for f in metrics.csv train.log; do aws s3 cp $D/$f $OUT/$f --only-show-errors 2>/dev/null; done
    aws s3 cp /var/log/f001.log $OUT/rotation.log --only-show-errors 2>/dev/null
  done ) &
SYNC=$!

while true; do
  done_e=$(epochs_done)
  [ $done_e -ge $TOTAL ] && { log "all $TOTAL epochs done"; break; }
  k=$((done_e / EPOCHS_PER_BLOCK)); stop_at=$(((k + 1) * EPOCHS_PER_BLOCK))
  if [ "$(cat $MARK)" != "$k" ]; then                  # rotate: put block k in the train folder
    # Only the prefetch: a bare `wait` would also wait on the endless S3 mirror loop.
    [ -n "${PREFETCH:-}" ] && wait $PREFETCH 2>/dev/null
    if [ "$(cat $STAGE_MARK 2>/dev/null)" = "$k" ] && [ $(ls $STAGE/*.tar 2>/dev/null | wc -l) -eq $PER_BLOCK ]; then
      rm -rf $TRAIN && mv $STAGE $TRAIN && rm -f $STAGE_MARK
    else
      log "block $k not staged - fetching it now"; rm -rf $TRAIN $STAGE $STAGE_MARK
      fetch_block $k $TRAIN || { log "fetch of block $k failed"; exit 1; }
    fi
    echo $k > $MARK; log "rotated to block $k (shards $((k * PER_BLOCK))-$((k * PER_BLOCK + PER_BLOCK - 1)))"
  fi
  nk=$((k + 1)); PREFETCH=""
  if [ $nk -lt $((TOTAL / EPOCHS_PER_BLOCK)) ] && [ "$(cat $STAGE_MARK 2>/dev/null)" != "$nk" ]; then
    rm -rf $STAGE; ( fetch_block $nk $STAGE && echo $nk > $STAGE_MARK && log "block $nk staged" ) & PREFETCH=$!
  fi
  log "training block $k: epochs $((done_e + 1))-$stop_at"
  for attempt in 1 2 3; do
    (cd /opt/allotrope && PYTHONPATH=. python scripts/train_foundation_model.py $D/config.json >> $D/train.log 2>&1) &
    T=$!
    while kill -0 $T 2>/dev/null; do
      if [ $(epochs_done) -ge $stop_at ]; then pkill -TERM -f "train_foundation_model.py $D/config.json"; break; fi
      sleep 30
    done
    wait $T
    [ $(epochs_done) -ge $stop_at ] && break
    log "trainer exited at epoch $(epochs_done) before $stop_at (attempt $attempt) - resuming"; sleep 60
  done
  [ $(epochs_done) -ge $stop_at ] || { log "block $k failed 3 times - stopping"; break; }
done
kill $SYNC 2>/dev/null
for f in metrics.csv train.log; do aws s3 cp $D/$f $OUT/$f --only-show-errors; done
aws s3 cp /var/log/f001.log $OUT/rotation.log --only-show-errors
log "F001 driver finished"
