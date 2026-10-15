#!/bin/bash
# Register the 79 AVIRIS-NG India flight lines (copied from Drive by copy_userdata.sh into the
# inventory bucket's staging prefix) as aviris_ng L2 scene datasets, one per flight line, via
# import-s3: every object is re-hashed and copied server-side (docs/lld/inventory.md). Resumable:
# lines in done.txt are skipped. Runs on the prod box; the API key comes from the PRISMA import script.
eval "$(grep '^export INVENTORY' /opt/prisma_import.sh)"
SRC=s3://allotrope-raw-inventory-india/_staging/aviris_ng/aviris_datasets/
LOG=/var/log/aviris-import; mkdir -p $LOG; touch $LOG/done.txt $LOG/failed.txt
n=0
for line in $(aws s3 ls $SRC | awk '/PRE ang/ {print $2}' | tr -d /); do
  n=$((n+1))
  grep -qx "$line" $LOG/done.txt && continue
  # ang20180318t060114 -> 2018-03-18T06:01:14Z; product version from the reflectance file (corr_v2s1 / corr_v2m2)
  t=${line#ang}; iso="${t:0:4}-${t:4:2}-${t:6:2}T${t:9:2}:${t:11:2}:${t:13:2}Z"
  ver=$(aws s3 ls $SRC$line/ | awk '{print $4}' | grep -o 'corr_v[0-9a-z]*' | head -1)
  echo "== $(date -u +%H:%M:%S) [$n] $line $iso $ver"
  (cd /opt/allotrope/backend && /opt/inv-cli/bin/python -m allotrope_inventory.cli import-s3 "$SRC$line/" \
      --sensor aviris_ng --level L2 --scene-id "$line" --kind scene --acquired-at "$iso" \
      --meta "product=$ver" --meta "campaign=AVIRIS-NG India (ISRO-NASA)" --meta "source=gdrive:PS_11/aviris_datasets" \
      --tag aviris-ng-india --tag gdrive-2026-10 --hash-workers 8 --copy-workers 4) > $LOG/last.log 2>&1
  rc=$?
  tr '\r' '\n' < $LOG/last.log | grep -v '^  hashed' | tail -2
  if [ $rc -eq 0 ]; then echo "$line" >> $LOG/done.txt; else echo "FAILED rc=$rc $line" | tee -a $LOG/failed.txt; tail -5 $LOG/last.log; fi
done
echo "== finished $(date -u +%FT%TZ): $(wc -l < $LOG/done.txt) done, $(wc -l < $LOG/failed.txt) failed" | tee $LOG/FINISHED
