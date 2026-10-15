#!/bin/bash
# EC2 user-data (2026-10-15): copy the AVIRIS-NG India flight lines from Google Drive
# (My Drive/PS_11/aviris_datasets, 79 lines, ~771 GB) into a staging prefix of the inventory bucket,
# verify every file against Drive's MD5, upload the logs, then power off. The instance is launched
# with shutdown behaviour "terminate", so powering off ends it - nothing is left running.
# The Drive token (read-only scope) is read from /allotrope/prod/rclone_gdrive_token.
# Next step, on the prod box: register each flight line with `import-s3` (docs/lld/inventory.md).
set -uo pipefail
exec > /var/log/aviris-copy.log 2>&1
STAGE=s3:allotrope-raw-inventory-india/_staging/aviris_ng/aviris_datasets
STATUS=s3://allotrope-raw-inventory-india/_staging/aviris_ng/_copy
log() { echo "$(date -u +%FT%TZ) $*"; }
finish() {   # finish <status>: record it next to the data, keep the logs, power off
  echo "$1 $(date -u +%FT%TZ)" > /tmp/STATUS
  for f in /tmp/STATUS /var/log/aviris-copy.log /var/log/rclone.log /var/log/rclone-check.log; do
    [ -f $f ] && aws s3 cp $f $STATUS/$(basename $f) --region ap-south-1 --only-show-errors; done
  shutdown -h +1
  exit 0
}

log "installing rclone v1.75.1"
dnf install -y -q unzip
cd /tmp && curl -sSfO https://downloads.rclone.org/v1.75.1/rclone-v1.75.1-linux-amd64.zip && unzip -q rclone-v1.75.1-linux-amd64.zip \
  && install -m 755 rclone-v1.75.1-linux-amd64/rclone /usr/local/bin/rclone || finish "failed-install"

TOKEN=$(aws ssm get-parameter --name /allotrope/prod/rclone_gdrive_token --with-decryption --query Parameter.Value --output text --region ap-south-1) || finish "failed-token"
mkdir -p /root/.config/rclone
cat > /root/.config/rclone/rclone.conf <<EOF
[gd]
type = drive
scope = drive.readonly
# PS_11 is a shared folder (a shortcut in My Drive), so paths do not resolve: address the
# aviris_datasets folder by the ID in the user's link (verified 2026-10-15: 79 flight lines).
root_folder_id = 1fm1LKEObAHstvE8Qeh0W4YEpMEqOCz_V
token = $TOKEN
[s3]
type = s3
provider = AWS
env_auth = true
region = ap-south-1
storage_class = INTELLIGENT_TIERING
EOF
chmod 600 /root/.config/rclone/rclone.conf
unset TOKEN

log "source size:"; rclone size gd: || finish "failed-list"
log "copying"
rclone copy gd: $STAGE --transfers 8 --checkers 16 --drive-chunk-size 64M \
  --s3-chunk-size 64M --s3-upload-concurrency 4 --retries 5 --low-level-retries 20 \
  --stats 60s --stats-one-line -v --log-file /var/log/rclone.log || finish "failed-copy"
log "verifying against Drive's MD5"
rclone check gd: $STAGE --one-way --log-file /var/log/rclone-check.log -v || finish "failed-check"
log "done"
rm -f /root/.config/rclone/rclone.conf
finish "ok"
