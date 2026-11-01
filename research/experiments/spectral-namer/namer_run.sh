#!/bin/bash
# Spectral-namer driver on the H100 capacity-block box (block 2, cr-0dccba00b34dc29e1, 2026-10-31). Mounts the NVMe,
# unpacks the code bundle, stages the fine-tune shard sets (EnMAP / PRISMA i16dz, AVIRIS-NG native) and the V2
# checkpoint, then runs each named config in turn (default N001 then N002):
#   setsid nohup bash /opt/namer_run.sh SHA N001 N002 > /var/log/namer.log 2>&1 < /dev/null &
# Mirrors metrics.csv / train.log / gpu.txt to S3 every 5 min and every epoch checkpoint at the end; each run writes
# EXIT (status + time). Re-running resumes: staged data is kept, the trainer resumes from last.pt.
set -uo pipefail
SHA=$1; shift; RUNS=${*:-N001 N002}
N=/opt/dlami/nvme; B=allotrope-training-data-prod-india; P=s3://$B/patches; X=s3://$B/experiments/indradhanu-finetune/spectral-namer   # inside the GPU role's write prefixes
log() { echo "$(date -u +%FT%TZ) $*"; }
if ! mountpoint -q $N; then
  DEV=$(lsblk -dpno NAME,MODEL | grep -i "instance storage" | head -1 | awk '{print $1}')
  mkfs.ext4 -q -F "$DEV" && mkdir -p $N && mount "$DEV" $N
fi
mkdir -p /opt/allotrope && cd /opt/allotrope && aws s3 cp s3://$B/deploy/allotrope-$SHA.tar.gz code.tgz --only-show-errors \
  && tar xzf code.tgz && echo $SHA > COMMIT || { log "no code bundle $SHA"; exit 1; }
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source /opt/pytorch/bin/activate
pip install --quiet webdataset "pydantic>=2" tqdm boto3 threadpoolctl
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p $N/models $N/ft/band_lists
[ -s $N/models/FTH3_ep5.pt ] || aws s3 cp s3://$B/experiments/indradhanu-finetune/FTH3-heavy-joint-more2/ckpt_all/spectral_stem_finetune_v0.5.0_epoch5.pt $N/models/FTH3_ep5.pt --only-show-errors
aws s3 cp --recursive s3://$B/experiments/indradhanu-finetune/band_lists/ $N/ft/band_lists/ --only-show-errors
if ! grep -q '^ok' $N/ft/STAGED 2>/dev/null; then
  for spec in "enmap enmap/native-v1z" "prisma prisma/native-v1z" "aviris_ng aviris_ng/native-v1"; do
    set -- $spec
    for split in test train; do
      log "pulling $1/$split"; mkdir -p $N/ft/$1/$split
      aws s3 ls $P/$2/$split/ | awk '/\.tar$/ {print $4}' | xargs -P 32 -I{} sh -c \
        "[ -s $N/ft/$1/$split/{} ] || aws s3 cp $P/$2/$split/{} $N/ft/$1/$split/{} --only-show-errors" \
        || { log "pull of $1/$split failed"; exit 1; }
    done
  done
  for d in $N/ft/*/train $N/ft/*/test; do log "$(ls $d/*.tar | wc -l) shards in $d"; done
  echo "ok $(date -u +%FT%TZ)" > $N/ft/STAGED
fi
df -h $N | tail -1
for RUN in $RUNS; do
  D=$N/runs/$RUN; mkdir -p $D
  [ -f $D/EXIT ] && grep -q '^0 ' $D/EXIT && { log "$RUN already done"; continue; }
  cp configs/namer/$RUN.json $D/config.json && aws s3 cp $D/config.json $X/$RUN/config.json --only-show-errors
  ( while true; do sleep 300
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu.txt
      for f in metrics.csv train.log gpu.txt; do aws s3 cp $D/$f $X/$RUN/$f --only-show-errors 2>/dev/null; done
      aws s3 cp $D/best.pt $X/$RUN/best.pt --only-show-errors 2>/dev/null
    done ) & SYNC=$!
  log "$RUN starting"
  PYTHONPATH=. python scripts/train_namer.py $D/config.json >> $D/train.log 2>&1
  STATUS=$?; kill $SYNC
  echo "$STATUS $(date -u +%FT%TZ)" > $D/EXIT
  for f in metrics.csv train.log EXIT best.pt last.pt; do aws s3 cp $D/$f $X/$RUN/$f --only-show-errors; done
  aws s3 cp --recursive $D/ $X/$RUN/ckpt_all/ --exclude "*" --include "ep*.pt" --only-show-errors
  log "$RUN exited $STATUS"
done
log "namer driver finished"
