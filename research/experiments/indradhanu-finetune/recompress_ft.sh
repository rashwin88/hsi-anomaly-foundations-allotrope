#!/bin/bash
# Stage 0 (2026-10-16): re-encode the EnMAP and PRISMA native shards to pixels.i16dz (exact, ~1.9x)
# so all three fine-tuning sets fit on the L40S's 412 GB NVMe. Runs on the prod box (in-region).
# AVIRIS-NG shards are already i16dz. DONE holds ok|failed-<set> and a timestamp.
B=s3://allotrope-training-data-prod-india/patches; L=/var/log/recompress-ft; mkdir -p $L; rm -f $L/DONE
RUN="docker run --rm --network host -v /opt/se-shard:/w -w /w -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/w -e AWS_DEFAULT_REGION=ap-south-1 se-recompress:1 python scripts/recompress_shards.py"
for spec in "enmap 97 7" "prisma 99 11"; do
  set -- $spec; S=$1; TR=$2; TE=$3
  for k in 0 1 2 3; do $RUN $B/$S/native-v1/ $B/$S/native-v1z/ --train $TR --test $TE --part $k/4 >> $L/${S}_$k.log 2>&1 & done; wait
  $RUN $B/$S/native-v1/ $B/$S/native-v1z/ --train $TR --test $TE --finalize > $L/${S}_finalize.log 2>&1 || { echo "failed-$S $(date -u +%FT%TZ)" > $L/DONE; exit 1; }
done
echo "ok $(date -u +%FT%TZ)" > $L/DONE
