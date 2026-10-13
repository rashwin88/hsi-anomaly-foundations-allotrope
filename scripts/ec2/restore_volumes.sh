#!/usr/bin/env bash
# Load the model checkpoints and splib07 caches into their named volumes
# from tarballs staged in S3 by the laptop (see docs/08-deploy.md). Run as
# root on the box before the first `up`. Refuses to overwrite a non-empty
# volume.
set -euo pipefail
src=s3://allotrope-training-data-prod-india/deploy/volumes

for vol in models splib07; do
    docker volume create "allotrope_$vol" >/dev/null
    if [ -n "$(docker run --rm -v "allotrope_$vol:/v" alpine ls -A /v)" ]; then
        echo "allotrope_$vol already populated; skipping"; continue
    fi
    aws s3 cp --region ap-south-1 --only-show-errors "$src/$vol.tar" - \
        | docker run --rm -i -v "allotrope_$vol:/v" alpine tar -x -C /v
    echo "allotrope_$vol: $(docker run --rm -v "allotrope_$vol:/v" alpine du -sh /v | cut -f1)"
done
