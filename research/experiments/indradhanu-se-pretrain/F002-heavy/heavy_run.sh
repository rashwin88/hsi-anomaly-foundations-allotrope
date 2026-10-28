#!/bin/bash
# F002-heavy (Indradhanu Heavy) driver on the H100 capacity-block box (2026-10-28). Mount NVMe, code, deps, then
# F001's rotation recipe over 3,200 train shards: 16 blocks of 200, 5 epochs each, 80 epochs. Re-running
# resumes (metrics.csv + BLOCK markers + the trainer's auto_resume from S3 last.pt), so block 2 reuses it as is.
#   setsid nohup bash /opt/heavy_run.sh > /var/log/heavy.log 2>&1 < /dev/null &
set -uo pipefail
SHA=7947bef
N=/opt/dlami/nvme; R=F002-heavy; D=$N/runs/$R
B=allotrope-training-data-prod-india; SHARDS=s3://$B/patches/spectralearth/se-v1
OUT=s3://$B/experiments/indradhanu-se-pretrain/$R
TRAIN=$N/se_cache/train/se-v1/w128_h128_s64; TEST=$N/se_cache/test/se-v1/w128_h128_s64; MARK=$N/se_cache/train/BLOCK
STAGE=$N/se_stage; STAGE_MARK=$N/se_stage.BLOCK
PER_BLOCK=200; EPOCHS_PER_BLOCK=5; TOTAL=80; NBLOCKS=$(( (TOTAL + EPOCHS_PER_BLOCK - 1) / EPOCHS_PER_BLOCK ))
log() { echo "$(date -u +%FT%TZ) $*"; }
if ! mountpoint -q $N; then
  DEV=$(lsblk -dpno NAME,MODEL | grep -i "instance storage" | head -1 | awk '{print $1}')
  mkfs.ext4 -q -F "$DEV" && mkdir -p $N && mount "$DEV" $N
fi
df -h $N; mkdir -p $D $TRAIN $TEST
mkdir -p /opt/allotrope && cd /opt/allotrope && aws s3 cp s3://$B/deploy/allotrope-$SHA.tar.gz code.tgz --only-show-errors && tar xzf code.tgz && echo $SHA > COMMIT
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source /opt/pytorch/bin/activate
pip install --quiet webdataset "pydantic>=2" tqdm boto3 threadpoolctl
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
aws s3 cp $OUT/config.json $D/config.json --only-show-errors || { log "no config"; exit 1; }
for f in metrics.csv train.log; do aws s3 cp $OUT/$f $D/$f --only-show-errors 2>/dev/null; done
aws s3 cp $SHARDS/stats.json $N/se_cache/stats.json --only-show-errors
seq 0 15 | xargs -P 16 -I{} sh -c 'f=$(printf "shard-%06d.tar" {}); [ -s '"$TEST"'/$f ] || aws s3 cp '"$SHARDS"'/test/$f '"$TEST"'/$f --only-show-errors'
epochs_done() { [ -f $D/metrics.csv ] && echo $(($(wc -l < $D/metrics.csv) - 1)) || echo 0; }
fetch_block() { local k=$1 dir=$2; mkdir -p $dir
  seq $((k * PER_BLOCK)) $((k * PER_BLOCK + PER_BLOCK - 1)) | xargs -P 32 -I{} sh -c     'f=$(printf "shard-%06d.tar" {}); [ -s '"$dir"'/$f ] || aws s3 cp '"$SHARDS"'/train/$f '"$dir"'/$f --only-show-errors'
  [ $(ls $dir/*.tar 2>/dev/null | wc -l) -eq $PER_BLOCK ]; }
( while true; do sleep 300
    for f in metrics.csv train.log; do aws s3 cp $D/$f $OUT/$f --only-show-errors 2>/dev/null; done
    aws s3 cp /var/log/heavy.log $OUT/rotation.log --only-show-errors 2>/dev/null
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu.txt; aws s3 cp $D/gpu.txt $OUT/gpu.txt --only-show-errors 2>/dev/null
  done ) & SYNC=$!
while true; do
  done_e=$(epochs_done); [ $done_e -ge $TOTAL ] && { log "all $TOTAL epochs done"; break; }
  k=$((done_e / EPOCHS_PER_BLOCK)); stop_at=$(((k + 1) * EPOCHS_PER_BLOCK)); [ $stop_at -gt $TOTAL ] && stop_at=$TOTAL
  if [ "$(cat $MARK 2>/dev/null)" != "$k" ]; then
    [ -n "${PREFETCH:-}" ] && wait $PREFETCH 2>/dev/null
    if [ "$(cat $STAGE_MARK 2>/dev/null)" = "$k" ] && [ $(ls $STAGE/*.tar 2>/dev/null | wc -l) -eq $PER_BLOCK ]; then
      rm -rf $TRAIN && mv $STAGE $TRAIN && rm -f $STAGE_MARK
    else log "block $k not staged - fetching"; rm -rf $TRAIN $STAGE $STAGE_MARK; fetch_block $k $TRAIN || { log "fetch $k failed"; exit 1; }; fi
    echo $k > $MARK; log "rotated to block $k"
  fi
  nk=$((k + 1)); PREFETCH=""
  if [ $nk -lt $NBLOCKS ] && [ "$(cat $STAGE_MARK 2>/dev/null)" != "$nk" ]; then
    rm -rf $STAGE; ( fetch_block $nk $STAGE && echo $nk > $STAGE_MARK && log "block $nk staged" ) & PREFETCH=$!
  fi
  log "training block $k: epochs $((done_e + 1))-$stop_at"
  for attempt in 1 2 3; do
    (cd /opt/allotrope && PYTHONPATH=. python scripts/train_foundation_model.py $D/config.json >> $D/train.log 2>&1) & T=$!
    while kill -0 $T 2>/dev/null; do
      [ $(epochs_done) -ge $stop_at ] && { pkill -TERM -f "train_foundation_model.py $D/config.json"; break; }; sleep 30; done
    wait $T; [ $(epochs_done) -ge $stop_at ] && break
    log "trainer exited at epoch $(epochs_done) before $stop_at (attempt $attempt)"; sleep 60
  done
  [ $(epochs_done) -ge $stop_at ] || { log "block $k failed 3 times"; break; }
done
kill $SYNC 2>/dev/null; for f in metrics.csv train.log; do aws s3 cp $D/$f $OUT/$f --only-show-errors; done
log "F002-heavy driver finished"
