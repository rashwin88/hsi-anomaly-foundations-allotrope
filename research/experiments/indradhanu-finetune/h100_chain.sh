#!/bin/bash
# H100 block-1 chain (2026-10-28): stage fine-tune data now; after F002-heavy's 80 epochs: FT04 pseudo-labels,
# then FTH1 (Heavy fine-tune), then FTB1 (control at FT04 size). Each step logs to /var/log/h100_chain.log.
#   setsid nohup bash /opt/h100_chain.sh > /var/log/h100_chain.log 2>&1 < /dev/null &
set -uo pipefail
N=/opt/dlami/nvme; B=allotrope-training-data-prod-india; P=s3://$B/patches
log() { echo "$(date -u +%FT%TZ) $*"; }
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source /opt/pytorch/bin/activate
# 1. stage fine-tune data (network only; runs alongside pretraining)
if ! grep -q '^ok' $N/ft/STAGED 2>/dev/null; then
  mkdir -p $N/ft/band_lists
  aws s3 cp --recursive s3://$B/experiments/indradhanu-finetune/band_lists/ $N/ft/band_lists/ --only-show-errors
  for spec in "enmap enmap/native-v1z" "prisma prisma/native-v1z" "aviris_ng aviris_ng/native-v1"; do
    set -- $spec
    for split in train test; do
      log "pulling $1/$split"
      aws s3 cp --recursive $P/$2/$split/ $N/ft/$1/$split/ --exclude "*" --include "*.tar" --only-show-errors || { log "pull $1/$split failed"; exit 1; }
    done
  done
  for d in $N/ft/*/train $N/ft/*/test; do log "$(ls $d/*.tar | wc -l) shards in $d"; done
  echo "ok $(date -u +%FT%TZ)" > $N/ft/STAGED
fi
# 2. wait for F002-heavy
until grep -q "F002-heavy driver finished" /var/log/heavy.log 2>/dev/null; do sleep 120; done
E=$(( $(wc -l < $N/runs/F002-heavy/metrics.csv) - 1 ))
[ $E -ge 80 ] || { log "F002-heavy stopped at $E epochs - chain stops"; exit 1; }
log "F002-heavy done ($E epochs)"
# 3. pseudo-labels from FT04 (PRISMA, AVIRIS-NG train shards)
mkdir -p $N/models && aws s3 cp s3://$B/experiments/indradhanu-finetune/FT04-final-joint/checkpoints/best.pt $N/models/FT04.pt --only-show-errors
cd /opt/allotrope
PYTHONPATH=. python scripts/pseudo_label.py $N/models/FT04.pt $N/ft/prisma/train $N/ft/pseudo_ft04/prisma --batch 64 || { log "pseudo prisma failed"; exit 1; }
PYTHONPATH=. python scripts/pseudo_label.py $N/models/FT04.pt $N/ft/aviris_ng/train $N/ft/pseudo_ft04/aviris_ng --band-list $N/ft/band_lists/aviris_ng_372.json --batch 64 || { log "pseudo aviris failed"; exit 1; }
log "pseudo-labels: $(ls $N/ft/pseudo_ft04/prisma | wc -l) prisma, $(ls $N/ft/pseudo_ft04/aviris_ng | wc -l) aviris_ng"
# 4. FTH1 then FTB1
for run in FTH1-heavy-joint FTB1-base-joint; do
  log "start $run"; bash /opt/ft_run.sh $run > /var/log/ft_$run.log 2>&1; log "$run: $(cat $N/runs/$run/EXIT 2>/dev/null)"
done
log "chain finished"
