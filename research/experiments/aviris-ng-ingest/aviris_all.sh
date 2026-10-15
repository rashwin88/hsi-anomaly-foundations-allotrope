#!/bin/bash
# AVIRIS-NG native-band shards, full run (2026-10-15): waits for the inventory import
# (/opt/aviris_import.sh) to finish with no failures, then plan -> tiles (2 parts) -> split ->
# shards (2 parts) -> finalize. Each step is resumable; DONE holds ok|failed-<step> and a timestamp.
O=s3://allotrope-training-data-prod-india/patches/aviris_ng/native-v1/
W=/opt/aviris-work; L=/var/log/aviris-shard; mkdir -p $W $L; rm -f $L/DONE
RUN="docker run --rm --network host -v /opt/se-shard:/w -v $W:$W -w /w -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/w -e AWS_DEFAULT_REGION=ap-south-1 se-recompress:1 python scripts/shard_aviris.py"
retry() { for a in 1 2 3 4 5; do "$@" && return 0; echo "ATTEMPT $a FAILED $(date -u +%FT%TZ)"; sleep 60; done; return 1; }
fail() { echo "failed-$1 $(date -u +%FT%TZ)" > $L/DONE; exit 1; }
until [ -f /var/log/aviris-import/FINISHED ]; do sleep 60; done
[ -s /var/log/aviris-import/failed.txt ] && fail import
date -u +%FT%TZ > $L/STARTED
[ -f $W/plan.json ] || retry $RUN plan s3://allotrope-raw-inventory-india/aviris_ng/L2/ $W > $L/plan.log 2>&1 || fail plan
if [ ! -d $W/tiles/train ]; then
  for k in 0 1; do ( retry $RUN tiles - $W --part $k/2 >> $L/tiles_$k.log 2>&1 ) & done; wait
  $RUN split - $W > $L/split.log 2>&1 || fail split
fi
for k in 0 1; do ( retry $RUN shards $W $O --part $k/2 >> $L/shards_$k.log 2>&1 ) & done; wait
$RUN finalize $W $O > $L/finalize.log 2>&1 || fail finalize
echo "ok $(date -u +%FT%TZ)" > $L/DONE
