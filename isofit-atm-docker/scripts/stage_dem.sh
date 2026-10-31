#!/usr/bin/env bash
# =============================================================================
# Stage Copernicus GLO-30 DEM tiles for a bounding box (or list of L1C scenes).
#
# Run on the CONNECTED node. Transfer the resulting directory to the air-gap
# host and mount at /aux/dem/ when running the pipeline.
#
# Source: AWS Open Data bucket (no auth required)
#         s3://copernicus-dem-30m/
# =============================================================================
set -euo pipefail

MODE="${1:-}"

usage() {
    cat <<EOF
Usage:
    $0 bbox  <min_lon> <min_lat> <max_lon> <max_lat> <output_dir>
    $0 scene <l1c_dir> [<l1c_dir> ...] <output_dir>

Requirements: gdal (gdalwarp, gdalbuildvrt), python3 with lxml, curl.
EOF
    exit 1
}

[[ -z "$MODE" ]] && usage

# Collect tile identifiers (1x1 deg, named e.g. N45_E011)
declare -a TILES=()

if [[ "$MODE" == "bbox" ]]; then
    [[ $# -eq 6 ]] || usage
    MIN_LON=$2; MIN_LAT=$3; MAX_LON=$4; MAX_LAT=$5; OUT_DIR=$6
    for lat in $(seq $((MIN_LAT - 1)) $((MAX_LAT + 1))); do
        for lon in $(seq $((MIN_LON - 1)) $((MAX_LON + 1))); do
            hemi_lat=$([ $lat -ge 0 ] && echo N || echo S)
            hemi_lon=$([ $lon -ge 0 ] && echo E || echo W)
            printf -v tile "%s%02d_%s%03d" "$hemi_lat" "${lat#-}" "$hemi_lon" "${lon#-}"
            TILES+=("$tile")
        done
    done
    mkdir -p "$OUT_DIR"

elif [[ "$MODE" == "scene" ]]; then
    OUT_DIR="${@: -1}"
    scenes=("${@:2:$#-2}")
    mkdir -p "$OUT_DIR"
    for scene in "${scenes[@]}"; do
        xml=$(ls "$scene"/*METADATA.XML 2>/dev/null | head -n1)
        [[ -z "$xml" ]] && { echo "no METADATA.XML in $scene"; continue; }
        # Extract scene corners with python
        eval "$(python3 - <<PY
import sys, re
from xml.etree import ElementTree as ET
t = ET.parse("$xml"); r = t.getroot()
for e in r.iter(): e.tag = re.sub(r'^\{.*\}', '', e.tag)
lats, lons = [], []
for pt in r.iter('point'):
    lat = pt.find('latitude'); lon = pt.find('longitude')
    if lat is not None and lon is not None:
        lats.append(float(lat.text)); lons.append(float(lon.text))
if not lats:  # try alternate schema
    for e in r.iter():
        if e.tag.lower() in ('centerlatitude','cornerlatitude'):
            lats.append(float(e.text))
        if e.tag.lower() in ('centerlongitude','cornerlongitude'):
            lons.append(float(e.text))
print(f"MIN_LAT={min(lats):.0f}; MAX_LAT={max(lats):.0f}; MIN_LON={min(lons):.0f}; MAX_LON={max(lons):.0f}")
PY
)"
        for lat in $(seq $((MIN_LAT - 1)) $((MAX_LAT + 1))); do
            for lon in $(seq $((MIN_LON - 1)) $((MAX_LON + 1))); do
                hemi_lat=$([ $lat -ge 0 ] && echo N || echo S)
                hemi_lon=$([ $lon -ge 0 ] && echo E || echo W)
                printf -v tile "%s%02d_%s%03d" "$hemi_lat" "${lat#-}" "$hemi_lon" "${lon#-}"
                TILES+=("$tile")
            done
        done
    done
else
    usage
fi

# Deduplicate
readarray -t TILES < <(printf '%s\n' "${TILES[@]}" | sort -u)
echo "==> Will fetch ${#TILES[@]} unique tiles into $OUT_DIR"

# Copernicus DEM naming on AWS:
#   s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N45_00_E011_00_DEM/
#     Copernicus_DSM_COG_10_N45_00_E011_00_DEM.tif
BASE_URL="https://copernicus-dem-30m.s3.amazonaws.com"

for tile in "${TILES[@]}"; do
    # tile is e.g. N45_E011 -> reformat to N45_00_E011_00
    prefix="${tile:0:3}_00_${tile:4}_00"
    fname="Copernicus_DSM_COG_10_${prefix}_DEM.tif"
    dst="${OUT_DIR}/${fname}"
    if [[ -f "$dst" ]]; then
        echo "    have $fname"
        continue
    fi
    url="${BASE_URL}/Copernicus_DSM_COG_10_${prefix}_DEM/${fname}"
    echo "    fetching $fname"
    if ! curl -fSL --retry 3 -o "$dst" "$url"; then
        echo "    (missing tile - ocean or unavailable, skipping)"
        rm -f "$dst"
    fi
done

echo "==> Building VRT mosaic"
find "$OUT_DIR" -maxdepth 1 -name "Copernicus_DSM_COG_*.tif" > "${OUT_DIR}/.tiles.txt"
gdalbuildvrt -input_file_list "${OUT_DIR}/.tiles.txt" "${OUT_DIR}/cop30_mosaic.vrt"
rm "${OUT_DIR}/.tiles.txt"

echo
echo "==> DEM staged in: $OUT_DIR"
echo "    Mosaic: ${OUT_DIR}/cop30_mosaic.vrt"
echo "    Transfer this whole directory to the air-gap host and mount at /aux/dem/"
