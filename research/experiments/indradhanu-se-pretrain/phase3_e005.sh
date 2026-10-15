#!/bin/bash
# Phase 3 (decided 2026-10-15): extend E005 from 10 to 20 epochs. auto_resume continues from its own
# last.pt (epoch 10) with optimizer, scheduler and early-stop state; metrics.csv is appended to.
set -uo pipefail
N=/opt/dlami/nvme; R=E005-band-robust; D=$N/runs/$R; LOG=/var/log/run-next-e005b.log
pgrep -f train_foundation_model >/dev/null && { echo "training already running"; exit 1; }
[ -f $D/ckpt/last.pt ] || { echo "no local last.pt"; exit 1; }
mv $D/EXIT $D/EXIT.epoch10; mv $D/CAPPED_AT_10 $D/CAPPED_AT_10.done 2>/dev/null
echo "$(date -u +%FT%TZ) extending $R to 20 epochs (code: $(readlink /opt/allotrope))"
(setsid nohup bash /opt/run_next.sh $R > $LOG 2>&1 < /dev/null &)
sleep 60
while true; do
  if [ $(($(wc -l < $D/metrics.csv) - 1)) -ge 20 ] && pgrep -f train_foundation_model >/dev/null; then
    echo "$(date -u +%FT%TZ) $R reached 20 epochs - stopping" | tee -a $D/CAPPED_AT_20
    pkill -TERM -f train_foundation_model; sleep 120
  fi
  grep -q 'queue done' $LOG && { echo "$(date -u +%FT%TZ) $R extension done"; exit 0; }
  sleep 30
done
