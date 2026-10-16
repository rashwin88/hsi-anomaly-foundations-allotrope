#!/bin/bash
# Run one fine-tuning stage on the kept L40S (prepared 2026-10-16; run only on the user's go):
#   setsid nohup bash /opt/ft_run.sh FT01-probe > /var/log/ft_FT01-probe.log 2>&1 < /dev/null &
# Needs ft_stage.sh's STAGED=ok and the code bundle at /opt/allotrope. Mirrors metrics, the per-sensor /
# IoU table and the log to S3 every 5 minutes; checkpoints go to S3 from the trainer (auto_resume picks
# up after an interruption). Writes EXIT with the trainer's status.
set -uo pipefail
RUN=$1; N=/opt/dlami/nvme; D=$N/runs/$RUN; OUT=s3://allotrope-training-data-prod-india/experiments/indradhanu-finetune/$RUN
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source /opt/pytorch/bin/activate
grep -q '^ok' $N/ft/STAGED 2>/dev/null || { echo "data not staged"; exit 1; }
mkdir -p $D && aws s3 cp $OUT/config.json $D/config.json --only-show-errors || { echo "no config in S3"; exit 1; }
for f in metrics.csv finetune_metrics.csv train.log; do aws s3 cp $OUT/$f $D/$f --only-show-errors 2>/dev/null; done
( while true; do sleep 300; for f in metrics.csv finetune_metrics.csv train.log; do aws s3 cp $D/$f $OUT/$f --only-show-errors 2>/dev/null; done; done ) &
SYNC=$!
(cd /opt/allotrope && PYTHONPATH=. python scripts/train_foundation_model.py $D/config.json >> $D/train.log 2>&1)
STATUS=$?; kill $SYNC
echo "$STATUS $(date -u +%FT%TZ)" > $D/EXIT
for f in metrics.csv finetune_metrics.csv train.log EXIT; do aws s3 cp $D/$f $OUT/$f --only-show-errors; done
echo "$RUN exited $STATUS"
