#!/bin/bash
# Phase 2 (decided 2026-10-14): once the E004/E001/E002 queue is done, run E005 on the new code
# (bundle aebb95b: band-list fixes, course part 42), capped at 10 epochs. The queue's code is kept.
set -uo pipefail
N=/opt/dlami/nvme; B=allotrope-training-data-prod-india; SHA=aebb95b; LOG=/var/log/run-next-e005.log
echo "$(date -u +%FT%TZ) waiting for the E004/E001/E002 queue"
until grep -q 'queue done' /var/log/run-next.log; do sleep 60; done
while pgrep -f train_foundation_model >/dev/null; do sleep 30; done
mkdir -p /opt/allotrope-e005-$SHA && cd /opt/allotrope-e005-$SHA
aws s3 cp s3://$B/deploy/allotrope-e005-$SHA.tar.gz code.tgz --only-show-errors && tar xzf code.tgz || { echo "bundle fetch failed"; exit 1; }
grep -q band_widths app/foundation_models/components/spectral_reduced.py || { echo "bundle lacks the fixes"; exit 1; }
if [ -L /opt/allotrope ]; then rm /opt/allotrope; else mv /opt/allotrope /opt/allotrope-queue; fi
ln -s /opt/allotrope-e005-$SHA /opt/allotrope
echo "$(date -u +%FT%TZ) code switched to $SHA; starting E005"
(setsid nohup bash /opt/run_next.sh E005-band-robust > $LOG 2>&1 < /dev/null &)
sleep 60
while true; do
  M=$N/runs/E005-band-robust/metrics.csv
  if [ -f $M ] && [ $(($(wc -l < $M) - 1)) -ge 10 ] && pgrep -f train_foundation_model >/dev/null; then
    echo "$(date -u +%FT%TZ) E005-band-robust reached 10 epochs - stopping" | tee -a $N/runs/E005-band-robust/CAPPED_AT_10
    pkill -TERM -f train_foundation_model; sleep 120
  fi
  grep -q 'queue done' $LOG && { echo "$(date -u +%FT%TZ) E005 queue done"; exit 0; }
  sleep 30
done
