#!/usr/bin/env bash
# =============================================================================
# EnMAP L1C -> L2A pipeline entrypoint.
#
# Usage inside container:
#   /opt/pipeline/entrypoint.sh <l1c_dir> <l2a_dir> [--season summer|winter|auto]
#                                                   [--n-cores N]
#                                                   [--keep-work]
#
# Environment variables consumed:
#   AUX_ROOT       (default /opt/aux)   - baked-in aux data
#   CONFIG_ROOT    (default /opt/pipeline/configs)
#   DEM_VRT        (default /aux/dem/cop30_mosaic.vrt) - mounted DEM
#   N_CORES        (default: nproc)
# =============================================================================

set -euo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# ---- Argument parsing -------------------------------------------------------
if [[ $# -lt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
EnMAP L1C -> L2A (ISOFIT + sRTMnet)

Usage:
  entrypoint.sh <l1c_input_dir> <l2a_output_dir> [options]

Options:
  --season {summer|winter|auto}   Season preset (default: auto from acq month)
  --n-cores N                     Parallel workers (default: all CPUs)
  --line {analytical|empirical}   Per-pixel solve (default: analytical)
  --keep-work                     Keep intermediate ISOFIT working dir
  --stop-after-lut                Stop once the full LUT is built; keeps work dir
  -h, --help                      This message

Environment:
  DEM_VRT       Path to Copernicus DEM mosaic (mounted at /aux/dem/)
EOF
    exit 0
fi

INPUT_DIR="$1"; shift
OUTPUT_DIR="$1"; shift

SEASON="auto"
LINE_MODE="analytical"
N_CORES="${N_CORES:-$(nproc)}"
KEEP_WORK=0
STOP_AFTER_LUT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --season)    SEASON="$2"; shift 2 ;;
        --line)      LINE_MODE="$2"; shift 2 ;;
        --n-cores)   N_CORES="$2"; shift 2 ;;
        --keep-work) KEEP_WORK=1; shift ;;
        --stop-after-lut) STOP_AFTER_LUT=1; KEEP_WORK=1; shift ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---- Preflight --------------------------------------------------------------
[[ -d "$INPUT_DIR" ]] || die "Input directory not found: $INPUT_DIR"
mkdir -p "$OUTPUT_DIR"

DEM="${DEM_VRT:-/aux/dem/cop30_mosaic.vrt}"
[[ -f "$DEM" ]] || die "DEM not found: $DEM (mount your Cop-DEM VRT at /aux/dem/)"

# sRTMnet is now fetched by isofit itself into /root/.isofit/srtmnet
EMU=$(find /root/.isofit/srtmnet -name "*.h5" | head -n1)
[[ -n "$EMU" && -f "$EMU" ]] || die "sRTMnet .h5 not found under /root/.isofit/srtmnet"

# 6S binary must be present or sRTMnet's preSim step produces all-NaN LUTs.
# isofit/6S builds sixsV2.1.2.CO2 (plus a -lutaero variant), not sixsV2.1.
SIXS_EXE=$(find /root/.isofit/sixs -maxdepth 1 -name 'sixsV*' ! -name '*lutaero*' -type f -perm -u+x 2>/dev/null | head -n1)
[[ -n "$SIXS_EXE" ]] || die "6S binary missing under /root/.isofit/sixs (expected sixsV*)"
log "6S binary: $SIXS_EXE"

# Real surface priors only. The *_model_discrepancy / *_resid_abs .mat files
# in isofit-data are instrument correction files, NOT surface priors.
SURFACE=$(find "${AUX_ROOT}/surface" "${AUX_ROOT}/isofit-data" \
    \( -name "*surface*.mat" -o -name "*multicomponent*.mat" \) \
    2>/dev/null | head -n1 || true)
if [[ -n "$SURFACE" && -f "$SURFACE" ]]; then
    log "Using surface prior: $SURFACE"
    SURFACE_ARG="--surface_path=${SURFACE}"
else
    log "No pre-built surface prior found; ISOFIT will use built-in default"
    SURFACE_ARG=""
fi

# Explicit noise model. On 3.7 this maps to instrument_noise_path override in
# template_construction (line ~290). Belt-and-braces even if enmap sensor now
# has its own default in 3.7.
NOISE_ARG=""
if [[ -f /root/.isofit/data/avirisng_noise.txt ]]; then
    NOISE_ARG="--instrument_noise_path=/root/.isofit/data/avirisng_noise.txt"
    log "Using noise model: /root/.isofit/data/avirisng_noise.txt"
fi

# Channelized uncertainty (supplementary radiometric uncertainty per band)
CHAN_UNC_ARG=""
if [[ -f /root/.isofit/data/avirisc_noise.txt ]]; then
    CHAN_UNC_ARG="--channelized_uncertainty_path=/root/.isofit/data/avirisc_noise.txt"
fi

# empirical_line is patched at build time (see Dockerfile); analytical is
# the path upstream maintains, so it is the default.
case "$LINE_MODE" in
    analytical) LINE_ARG="--analytical_line" ;;
    empirical)  LINE_ARG="--empirical_line" ;;
    *) die "--line must be analytical or empirical, got: $LINE_MODE" ;;
esac
log "Per-pixel solve: $LINE_MODE"

log "Input:   $INPUT_DIR"
log "Output:  $OUTPUT_DIR"
log "DEM:     $DEM"
log "Cores:   $N_CORES"

WORK="${OUTPUT_DIR}/_work"
rm -rf "${WORK}/isofit"
mkdir -p "$WORK"

# ---- Stage 1: EnMAP L1C -> ISOFIT input cubes -------------------------------
log "Stage 1/3: preprocessing EnMAP L1C -> radiance/location/observation cubes"
python -m enmap_l1c_preprocess \
    --input   "$INPUT_DIR" \
    --dem     "$DEM" \
    --workdir "$WORK"

# ISOFIT apply_oe's template_construction parses input filename and expects
# at least 6 underscore-separated tokens. Rename our cubes to embed the
# EnMAP scene ID so split("_")[5] yields the acquisition timestamp.
XML_NAME=$(basename "$(ls "$INPUT_DIR"/*METADATA.XML | head -n1)")
SCENE_ID="${XML_NAME%-METADATA.XML}"
log "Renaming cubes with scene ID: $SCENE_ID"
for name in rdn loc obs; do
    if [[ -f "${WORK}/${name}" ]]; then
        mv "${WORK}/${name}"     "${WORK}/${SCENE_ID}_${name}"
        mv "${WORK}/${name}.hdr" "${WORK}/${SCENE_ID}_${name}.hdr"
    fi
done

# Resolve season if auto
if [[ "$SEASON" == "auto" ]]; then
    SEASON=$(cat "${WORK}/season.txt")
    log "Season auto-detected: $SEASON"
fi

SEASON_CFG="${CONFIG_ROOT}/enmap_${SEASON}.json"
[[ -f "$SEASON_CFG" ]] || die "No season config: $SEASON_CFG"

# ---- Stage 1b: surface prior on this scene's wavelength grid ----------------
# The baked prior uses a nominal EnMAP grid; per-scene centres drift enough
# that ISOFIT's check_surface_model warns. Rebuild on the real grid.
# ISOFIT 3.7 names surface states "RFL_%04i" % int(wl), and construct_full_state
# de-duplicates them through set(). EnMAP's VNIR/SWIR overlap puts two band
# pairs inside the same integer nm, so the output statevector ends up shorter
# than the forward model's and every write raises
# "len(fm.statevec) > len(self.full_statevec)". Nudge colliding bands across the
# integer boundary - at most 0.15 nm against a ~6.5 nm FWHM, and for the prior
# only; the radiance cube keeps its true band centres.
python - "$WORK" <<'PY'
import sys
import numpy as np
work = sys.argv[1]
q = np.loadtxt(f"{work}/wavelengths.txt")
wl = q[:, 1]  # a view: edits below land in q
for i in range(1, len(wl)):
    if int(wl[i]) == int(wl[i - 1]):
        if wl[i - 1] - int(wl[i - 1]) < int(wl[i]) + 1 - wl[i]:
            wl[i - 1] = int(wl[i - 1]) - 0.001
        else:
            wl[i] = int(wl[i]) + 1.001
assert len({int(w) for w in wl}) == len(wl), "wavelength de-collision failed"
np.savetxt(f"{work}/wavelengths_surface.txt", q, fmt="%d\t%.4f\t%.4f")
PY

SCENE_SURFACE="${WORK}/scene_surface.mat"
if isofit surface_model "${CONFIG_ROOT}/surface_multicomponent.json" \
        --wavelength_path "${WORK}/wavelengths_surface.txt" \
        --output_path     "$SCENE_SURFACE" \
        --seed 42 > "${WORK}/surface_model.log" 2>&1 && [[ -s "$SCENE_SURFACE" ]]; then
    log "Surface prior rebuilt on scene grid: $SCENE_SURFACE"
    SURFACE_ARG="--surface_path=${SCENE_SURFACE}"
else
    log "WARNING: per-scene surface_model failed (see ${WORK}/surface_model.log)"
    [[ -n "$SURFACE_ARG" ]] || die "no usable surface prior; --surface_path is required on 3.7"
fi

# ---- Stage 2: ISOFIT apply_oe -----------------------------------------------
log "Stage 2/3: ISOFIT apply_oe (sRTMnet emulator, ${LINE_MODE} line)"

run_oe() {
isofit apply_oe \
    "${WORK}/${SCENE_ID}_rdn" \
    "${WORK}/${SCENE_ID}_loc" \
    "${WORK}/${SCENE_ID}_obs" \
    "${WORK}/isofit" \
    enmap \
    --presolve \
    ${LINE_ARG} \
    --emulator_base="${EMU}" \
    --lut_config_file="${SEASON_CFG}" \
    ${SURFACE_ARG} \
    ${NOISE_ARG} \
    ${CHAN_UNC_ARG} \
    --n_cores="${N_CORES}" \
    --segmentation_size=50 \
    --pressure_elevation \
    --logging_level=INFO \
    2>&1 | tee "${WORK}/isofit.log"
}

# Diagnostic mode: apply_oe cannot stop itself after the LUT, so run it in the
# background and wait for the 4th "Loading LUT into memory" log line (presolve
# 6S + emulator, then full 6S + emulator). File size is no signal: ISOFIT
# pre-allocates lut.nc full-size and fills it in place. Exiting then works because this script is PID 1: the container, and
# every Ray worker in it, goes down with it. Killing apply_oe alone does not -
# orphaned workers hold the tee pipe open and the run hangs.
if [[ "$STOP_AFTER_LUT" -eq 1 ]]; then
    run_oe & OE_PID=$!
    until [[ $(grep -c "Loading LUT into memory" "${WORK}/isofit.log" 2>/dev/null || true) -ge 4 ]]; do
        kill -0 "$OE_PID" 2>/dev/null || die "apply_oe exited before the LUT was built"
        sleep 15
    done
    log "Stopped after LUT (${WORK}/isofit/lut_full). Work dir kept: $WORK"
    exit 0
fi
run_oe

# ---- Stage 3: repack as PACO-style L2A --------------------------------------
log "Stage 3/3: repacking ISOFIT output to PACO-style L2A layout"
python -m postprocess_to_paco \
    --isofit-dir "${WORK}/isofit" \
    --l1c-dir    "$INPUT_DIR" \
    --output     "$OUTPUT_DIR" \
    --season     "$SEASON" \
    --pipeline-version "${PIPELINE_VERSION:-unknown}"

# ---- Cleanup ---------------------------------------------------------------
if [[ "$KEEP_WORK" -eq 0 ]]; then
    log "Removing intermediate work dir (pass --keep-work to retain)"
    rm -rf "$WORK"
fi

log "DONE. L2A products in: $OUTPUT_DIR"
