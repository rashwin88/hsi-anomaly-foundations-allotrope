#!/bin/bash
# Queue further runs on the kept E003 instance: wait for the current run to finish, then run each
# named run in turn, reusing the code (/opt/allotrope) and shards (/opt/dlami/nvme/se_cache) that
# bootstrap.sh already put on the machine. Nothing is downloaded again except each run's config.
#
#   setsid nohup bash run_next.sh E001-linear E002-gate > /var/log/run-next.log 2>&1 < /dev/null &
#
# Per run: /opt/dlami/nvme/runs/<run>/{config.json,metrics.csv,train.log,EXIT,ckpt/}, mirrored to
# s3://allotrope-training-data-prod-india/experiments/indradhanu-se-pretrain/<run>/ every 5 minutes.
# A run that already has an EXIT file is skipped, so re-launching the queue after a reboot picks up
# where it stopped (and auto_resume continues a run that was cut off mid-way).
set -uo pipefail
B=allotrope-training-data-prod-india
N=/opt/dlami/nvme
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /opt/pytorch/bin/activate

echo "== $(date -u +%FT%TZ) waiting for E003 (EXIT file)"
while [ ! -f $N/run/EXIT ]; do sleep 60; done
echo "== $(date -u +%FT%TZ) E003 finished: $(cat $N/run/EXIT)"

for RUN in "$@"; do
  D=$N/runs/$RUN
  OUT=s3://$B/experiments/indradhanu-se-pretrain/$RUN
  mkdir -p $D
  if [ -f $D/EXIT ]; then echo "== $RUN already finished, skipping"; continue; fi
  aws s3 cp $OUT/config.json $D/config.json --only-show-errors || { echo "== $RUN: no config in S3, skipping"; continue; }
  aws s3 cp $OUT/metrics.csv $D/metrics.csv --only-show-errors 2>/dev/null
  aws s3 cp $OUT/train.log $D/train.log --only-show-errors 2>/dev/null
  echo "== $(date -u +%FT%TZ) starting $RUN"
  ( while true; do sleep 300
      aws s3 cp $D/metrics.csv $OUT/metrics.csv --only-show-errors 2>/dev/null
      aws s3 cp $D/train.log $OUT/train.log --only-show-errors 2>/dev/null
      nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu.txt
      aws s3 cp $D/gpu.txt $OUT/gpu.txt --only-show-errors 2>/dev/null
    done ) &
  SYNC=$!
  (cd /opt/allotrope && PYTHONPATH=. python scripts/train_foundation_model.py $D/config.json >> $D/train.log 2>&1)
  STATUS=$?
  kill $SYNC
  echo "$STATUS $(date -u +%FT%TZ)" > $D/EXIT
  for f in metrics.csv train.log EXIT; do aws s3 cp $D/$f $OUT/$f --only-show-errors; done
  echo "== $(date -u +%FT%TZ) $RUN exited $STATUS"
done
echo "== $(date -u +%FT%TZ) queue done"
