#!/bin/bash
# EC2 user-data for one run of the SpectralEarth pretraining experiment.
#
# Launched with __RUN__ (e.g. E003-cross-attention) and __SHA__ (code bundle commit) substituted.
# Runs once, on first boot: fetch code -> fetch a FIXED shard subset -> train -> upload -> stop.
# Everything a run produces lands in s3://allotrope-training-data-prod-india/experiments/
# indradhanu-se-pretrain/__RUN__/ (checkpoints every epoch via the trainer; metrics and logs every
# 5 minutes from here), so stopping or losing the instance loses nothing.
#
# Fixed subset, not the trainer's random.sample: train shards 000000-000199 (30,000 patches)
# and test 000000-000009 (1,500). The shard plan already shuffled patches across sites, so the
# first 200 are a random sample - and E001-E003 see exactly the same one. 200 because the
# smallest instance used (g6e.xlarge) has 250 GB of NVMe: 210 x 947 MiB = 194 GiB.
#
# The instance is never released (decided 2026-10-14, capacity was scarce): the script does not
# shut down, stop/terminate protection is on, and later runs reuse it via run_next.sh.
# Spot launches (launch.py --spot) terminate instead - spot cannot stop - which is fine:
# everything is in S3, and a relaunch resumes from checkpoints/last.pt.
set -uo pipefail
RUN=__RUN__
SHA=__SHA__
B=allotrope-training-data-prod-india
OUT=s3://$B/experiments/indradhanu-se-pretrain/$RUN
SHARDS=s3://$B/patches/spectralearth/se-v1
N=/opt/dlami/nvme
# Per-run folder, as in the E001+ configs and run_next.sh (E003, the first run, used $N/run).
D=$N/runs/$RUN
exec > >(tee -a /var/log/se-run.log) 2>&1
echo "== $(date -u +%FT%TZ) bootstrap $RUN @ $SHA"

# Instance store: the DLAMI mounts it at /opt/dlami/nvme; mount it ourselves if not.
if ! mountpoint -q $N; then
  DEV=$(lsblk -dpno NAME,MODEL | grep -i "instance storage" | head -1 | awk '{print $1}')
  mkfs.ext4 -q -F "$DEV" && mkdir -p $N && mount "$DEV" $N
fi
mkdir -p $D $N/ckpt $N/se_cache/train/se-v1/w128_h128_s64 $N/se_cache/test/se-v1/w128_h128_s64
df -h $N

# Code
mkdir -p /opt/allotrope && cd /opt/allotrope
aws s3 cp s3://$B/deploy/allotrope-$SHA.tar.gz code.tgz --only-show-errors && tar xzf code.tgz
# The DLAMI's activate script reads LD_LIBRARY_PATH unguarded; under `set -u` an unset one killed
# the first E003 boot (2026-10-14).
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
source /opt/pytorch/bin/activate
pip install --quiet webdataset "pydantic>=2" tqdm boto3
python -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
aws s3 cp $OUT/config.json $D/config.json --only-show-errors
# A relaunch after a spot interruption resumes from S3 (auto_resume): carry the metrics and
# log over too, or the 5-minute mirror below would overwrite them with only the new epochs.
aws s3 cp $OUT/metrics.csv $D/metrics.csv --only-show-errors 2>/dev/null && echo "resuming: $(($(wc -l < $D/metrics.csv) - 1)) epochs already in metrics.csv"
aws s3 cp $OUT/train.log $D/train.log --only-show-errors 2>/dev/null

# Data: the fixed subset, 16 downloads at a time
aws s3 cp $SHARDS/stats.json $N/se_cache/stats.json --only-show-errors
t0=$(date +%s)
{ for i in $(seq 0 199); do printf "train %06d\n" $i; done; for i in $(seq 0 9); do printf "test %06d\n" $i; done; } |
  xargs -P 16 -L 1 sh -c 'aws s3 cp '"$SHARDS"'/$0/shard-$1.tar '"$N"'/se_cache/$0/se-v1/w128_h128_s64/shard-$1.tar --only-show-errors'
echo "shards: $(ls $N/se_cache/train/se-v1/w128_h128_s64 | wc -l) train, $(ls $N/se_cache/test/se-v1/w128_h128_s64 | wc -l) test in $(( $(date +%s) - t0 )) s"

# Mirror metrics and logs to S3 every 5 minutes while training runs
( while true; do sleep 300
    aws s3 cp $D/metrics.csv $OUT/metrics.csv --only-show-errors 2>/dev/null
    aws s3 cp $D/train.log $OUT/train.log --only-show-errors 2>/dev/null
    aws s3 cp /var/log/se-run.log $OUT/bootstrap.log --only-show-errors 2>/dev/null
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > $D/gpu.txt
    aws s3 cp $D/gpu.txt $OUT/gpu.txt --only-show-errors 2>/dev/null
  done ) &
SYNC=$!

cd /opt/allotrope
PYTHONPATH=. python scripts/train_foundation_model.py $D/config.json >> $D/train.log 2>&1
STATUS=$?
echo "== $(date -u +%FT%TZ) training exited $STATUS"
kill $SYNC
echo "$STATUS $(date -u +%FT%TZ)" > $D/EXIT
for f in metrics.csv train.log EXIT; do aws s3 cp $D/$f $OUT/$f --only-show-errors; done
aws s3 cp /var/log/se-run.log $OUT/bootstrap.log --only-show-errors
# No shutdown: GPU capacity was scarce on 2026-10-14, so the instance is kept and reused for the
# next run (stop/terminate protection on; poweroff.target masked).
