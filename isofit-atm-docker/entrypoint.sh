#!/usr/bin/env bash
# =============================================================================
# L1 -> L2 surface reflectance (ISOFIT 4.1.5 + sRTMnet) for EnMAP, PRISMA,
# AVIRIS-NG, Tanager, GHOSt and unknown sensors, from native products, ENVI or
# GeoTIFF.
#
#   stage 1  readers.prep: read any delivery, fill time / location / geometry
#            (metadata -> derived -> assumed), decide the input level, write
#            ISOFIT's rdn / loc / obs and scene.json (rule, grade, checklist)
#   stage 2  ISOFIT apply_oe with the sensor's rule (noise, LUT, AOT prior);
#            skipped when the input is already surface reflectance
#   stage 2b atm_post (window mask, outliers, adjacency, terrain, snow), then
#            harmonisation with the rule's model
#   stage 3  writers
#
# Environment: CONFIG_ROOT (/opt/pipeline/configs), DEM_VRT (mounted DEM,
# optional: without it the run uses a flat 0 m and says so), N_CORES (nproc).
# =============================================================================

set -euo pipefail

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >&2; }
die() { log "ERROR: $*"; exit 1; }

if [[ $# -lt 2 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
L1 -> L2 surface reflectance (ISOFIT + sRTMnet), any supported sensor and format

Usage:
  entrypoint.sh <input file or folder> <output dir> [options]

Inputs: EnMAP L1C folder, PRISMA L1 .he5, AVIRIS-NG L1 folder (rdn/loc_ort/obs_ort),
Tanager .h5, or any sensor as ENVI (.hdr + data) or GeoTIFF.

Options:
  --sensor S              enmap|prisma|aviris_ng|tanager|ghost|unknown (default: identified)
  --input-level L         radiance|toa|sr: override the level tests (sr = pass through)
  --radiance-scale X      factor from delivered numbers to uW/cm^2/sr/nm (raw counts)
  --wavelength-file F     band centres [and FWHM] when the delivery has none
  --season S              summer|winter (default: from date and hemisphere)
  --no-harmonise          skip the harmonisation step
  --formats F             output formats, comma-separated: tif,bsq (default both)
  --paco-layout           EnMAP only: also write the PACO-style L2A folder
  --n-cores N             parallel workers (default: all CPUs)
  --line {analytical|empirical}   per-pixel solve (default: analytical)
  --ref PATH              reference product for the report's comparison section
                          (vendor L2 / FLAASH: EnMAP L2A folder, PRISMA L2 .he5,
                          Tanager SR .h5, GeoTIFF or ENVI)
  --ref-wavelength-file F band centres for a reference that carries none
  --no-report             skip the HTML quick-look report
  --keep-work             keep the intermediate work dir
  --stop-after-lut        stop once the full LUT is built; keeps the work dir
  -h, --help              this message

Grades and flags never stop a run; see scene.json in the work dir for what was
read, derived and assumed. Switches: SIXS_FAST=0 / SIXS_CACHE=0 (6S),
ATM_FAST_AL=0 (ISOFIT's own per-pixel solve).
EOF
    exit 0
fi

INPUT="$1"; shift
OUTPUT_DIR="$1"; shift
SEASON=""
LINE_MODE="analytical"
N_CORES="${N_CORES:-$(nproc)}"
KEEP_WORK=0
STOP_AFTER_LUT=0
HARMONISE=1
OUT_FORMATS="tif,bsq"
PACO=0
REPORT_HTML=1
REF_ARGS=()
PREP_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sensor|--input-level|--radiance-scale|--wavelength-file) PREP_ARGS+=("$1" "$2"); shift 2 ;;
        --season)         SEASON="$2"; shift 2 ;;
        --line)           LINE_MODE="$2"; shift 2 ;;
        --n-cores)        N_CORES="$2"; shift 2 ;;
        --no-harmonise)   HARMONISE=0; shift ;;
        --formats)        OUT_FORMATS="$2"; shift 2 ;;
        --paco-layout)    PACO=1; shift ;;
        --ref|--ref-wavelength-file) REF_ARGS+=("$1" "$2"); shift 2 ;;
        --no-report)      REPORT_HTML=0; shift ;;
        --keep-work)      KEEP_WORK=1; shift ;;
        --stop-after-lut) STOP_AFTER_LUT=1; KEEP_WORK=1; shift ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---- Preflight --------------------------------------------------------------
[[ -e "$INPUT" ]] || die "Input not found: $INPUT"
mkdir -p "$OUTPUT_DIR"
CONFIG_ROOT="${CONFIG_ROOT:-/opt/pipeline/configs}"
DEM="${DEM_VRT:-/aux/dem/cop30_mosaic.vrt}"
if [[ -f "$DEM" ]]; then PREP_ARGS+=(--dem "$DEM"); else
    log "WARNING: no DEM at $DEM: elevation 0 m assumed and flagged (mount a Cop-DEM VRT at /aux/dem/)"; DEM=""; fi
EMU=$(find /root/.isofit/srtmnet -name "*.h5" | head -n1)
[[ -n "$EMU" && -f "$EMU" ]] || die "sRTMnet .h5 not found under /root/.isofit/srtmnet"
case "$LINE_MODE" in
    analytical) LINE_ARG="--analytical_line" ;;   # empirical_line is patched at build time (Dockerfile)
    empirical)  LINE_ARG="--empirical_line" ;;
    *) die "--line must be analytical or empirical, got: $LINE_MODE" ;;
esac

WORK="${OUTPUT_DIR}/_work"
rm -rf "${WORK}/isofit"
mkdir -p "$WORK"
# 6S scattering cache shared across water-vapour nodes (build/sixs_wrapper.sh), fresh per run.
export SIXS_CACHE_DIR="${WORK}/sixs_cache"
rm -rf "$SIXS_CACHE_DIR"
# Stage start times for the report (stage_times.txt: "<stage> <epoch s>", one line per boundary).
: > "${WORK}/stage_times.txt"
stamp() { echo "$1 $(date +%s)" >> "${WORK}/stage_times.txt"; }
stamp read
log "Input:   $INPUT"
log "Output:  $OUTPUT_DIR"
log "Cores:   $N_CORES"
log "6S:      SIXS_FAST=${SIXS_FAST:-1} SIXS_CACHE=${SIXS_CACHE:-1}"
# apply_oe falls back to ISOFIT's own analytical line if fast_al does not import; say so up front.
log "Per-pixel solve: ${LINE_MODE}, ATM_FAST_AL=${ATM_FAST_AL:-1}, $(python -c 'import fast_al, torch; print("torch", torch.__version__, "on", "cuda" if torch.cuda.is_available() else "cpu")' 2>&1 | tail -1)"

# ---- Stage 1: read the delivery ----------------------------------------------
log "Stage 1/3: reading the delivery (readers.prep)"
if ! python -m readers.prep --input "$INPUT" --workdir "$WORK" "${PREP_ARGS[@]}" 2>&1 | tee "${WORK}/prep.log" >&2; then
    die "readers.prep stopped (see above; every stop names its override)"
fi

# scene.json -> shell variables
eval "$(python - "$WORK/scene.json" "$SEASON" <<'PY'
import json, shlex, sys
r = json.load(open(sys.argv[1])); u = r["rules"]
v = {"FID": r["fid"], "FMT": r["format"], "ISOFIT_SENSOR": r["isofit_sensor"], "RULE": r["rule"],
     "LEVEL": r["level"], "PASS": "1" if r["pass_through"] else "0", "GRADE": r["grade"],
     "NOISE": u["instrument_noise"] or "", "LUT_CFG": u["lut_config"] or "", "AOT_PRIOR": "1" if u["aot_prior"] else "0",
     "HARM_MODEL": u["harmonise_model"] or "", "SEASON": sys.argv[2] or r["season"],
     "LAT": "" if r["centre_lat"] is None else r["centre_lat"], "LON": "" if r["centre_lon"] is None else r["centre_lon"],
     "DATE": r["date"]}
print("\n".join(f"{k}={shlex.quote(str(x))}" for k, x in v.items()))
PY
)"
log "Scene:   $FID ($FMT), rule $RULE, ISOFIT sensor $ISOFIT_SENSOR, level $LEVEL, grade $GRADE"

if [[ "$PASS" == "1" ]]; then
    log "Input is surface reflectance (level tests or product label): no atmospheric correction."
    log "  Pass --input-level toa (or radiance) to correct it anyway."
    RFL="${WORK}/${FID}_rfl"
    REPORT=""
else
    # ---- Stage 1b: surface prior on the scene's exact band centres -----------
    # (AC26-EOE: a prior on shifted centres makes forward.upsample interpolate on
    # every evaluation, 40% of EnMAP's inversion time.)
    stamp surface_prior_and_lut_setup
    SCENE_SURFACE="${WORK}/scene_surface.mat"
    isofit surface_model "${CONFIG_ROOT}/surface_multicomponent.json" \
        --wavelength_path "${WORK}/wavelengths.txt" --output_path "$SCENE_SURFACE" \
        --seed 42 > "${WORK}/surface_model.log" 2>&1 && [[ -s "$SCENE_SURFACE" ]] \
        || die "surface_model failed (see ${WORK}/surface_model.log)"

    # ---- LUT preset, AOT prior, noise from the sensor's rule -----------------
    if [[ -n "$LUT_CFG" ]]; then LUT="${CONFIG_ROOT}/${LUT_CFG}"; else LUT="${CONFIG_ROOT}/enmap_${SEASON}.json"; fi
    [[ -f "$LUT" ]] || die "No LUT config: $LUT"
    AOT_ARGS=()
    if [[ "$AOT_PRIOR" == "1" ]]; then
        if [[ -n "$LAT" ]]; then
            python -m aot_prior --sensor "$RULE" --lat "$LAT" --lon "$LON" --date "$DATE" --base-lut "$LUT" \
                --out "${WORK}/lut.json" --record "${WORK}/aot_prior.json"
            LUT="${WORK}/lut.json"; AOT_ARGS=(--aot-prior "${WORK}/aot_prior.json")
        else
            log "WARNING: rule $RULE wants the climatology AOT prior but the scene has no location: free AOT"
        fi
    fi
    NOISE_ARG=()
    [[ -n "$NOISE" ]] && NOISE_ARG=(--instrument_noise_path="/root/.isofit/data/${NOISE}")
    log "Rule $RULE: LUT $(basename "$LUT"), noise ${NOISE:-ISOFIT default}, AOT prior $AOT_PRIOR"

    # ---- Stage 2: ISOFIT apply_oe --------------------------------------------
    # In its generic NA- mode ISOFIT takes the file id from the radiance file's stem, then names everything
    # fid + "_rdn" with "_rdn" swapped for "_lbl", "_rfl", ...: a stem that already ends in "_rdn" gets swapped
    # twice ("x_lbl_lbl") and the analytical line cannot find its segmentation file. So hand it a link named
    # after the bare file id. Sensor modes (enmap, prisma, ang, tanager) cut the id before "_rdn" themselves.
    RDN_IN="${WORK}/${FID}_rdn"
    if [[ "$ISOFIT_SENSOR" == NA-* ]]; then
        ln -sfn "${FID}_rdn" "${WORK}/${FID}"; ln -sfn "${FID}_rdn.hdr" "${WORK}/${FID}.hdr"
        RDN_IN="${WORK}/${FID}"
    fi
    log "Stage 2/3: ISOFIT apply_oe (sRTMnet emulator, ${LINE_MODE} line)"
    stamp isofit
    run_oe() {
    isofit apply_oe \
        "$RDN_IN" "${WORK}/${FID}_loc" "${WORK}/${FID}_obs" "${WORK}/isofit" "$ISOFIT_SENSOR" \
        --presolve ${LINE_ARG} --emulator_base="${EMU}" --lut_config_file="${LUT}" \
        --surface_path="${SCENE_SURFACE}" "${NOISE_ARG[@]}" --n_cores="${N_CORES}" --segmentation_size=50 \
        --pressure_elevation --logging_level=INFO \
        2>&1 | tee "${WORK}/isofit.log"
    }
    # Diagnostic mode: apply_oe cannot stop itself after the LUT, so run it in the
    # background and wait for the 4th "Running any post-sim functions" log line
    # (presolve 6S + emulator, then full 6S + emulator). Exiting then works because
    # this script is PID 1: the container, and every Ray worker in it, goes down with it.
    if [[ "$STOP_AFTER_LUT" -eq 1 ]]; then
        run_oe & OE_PID=$!
        until [[ $(grep -c "Running any post-sim functions" "${WORK}/isofit.log" 2>/dev/null || true) -ge 4 ]]; do
            kill -0 "$OE_PID" 2>/dev/null || die "apply_oe exited before the LUT was built"
            sleep 15
        done
        log "Stopped after LUT (${WORK}/isofit/lut_full). Work dir kept: $WORK"
        exit 0
    fi
    run_oe || die "apply_oe failed (see ${WORK}/isofit.log)"

    # ---- Stage 2b: per-sensor post-processing, in place on ISOFIT's rfl ------
    stamp post_process
    PX_M=$(python -c "import json; print(json.load(open('${WORK}/scene.json')).get('pixel_m') or 30)")
    log "Stage 2b/3: post-processing (window mask, outliers, adjacency, terrain, snow), rule $RULE"
    python -m atm_post --sensor "$RULE" --isofit-dir "${WORK}/isofit" --obs "${WORK}/${FID}_obs" --px-m "$PX_M" \
        "${AOT_ARGS[@]}"
    REPORT="${WORK}/isofit/atm_post.json"
    RFL=$(python -c "import json; print(json.load(open('$REPORT'))['reflectance'])")
fi

# ---- Stage 2c: harmonisation with the rule's model ---------------------------
if [[ "$HARMONISE" -eq 1 && -n "$HARM_MODEL" && -n "$REPORT" ]]; then
    stamp harmonise
    log "Stage 2c/3: harmonisation (${HARM_MODEL})"
    python -m harmonise "${CONFIG_ROOT}/harmonise/${HARM_MODEL}" "$REPORT" "${WORK}/${FID}_obs"
    RFL="${RFL}_harm"
else
    log "Stage 2c/3: harmonisation skipped (rule $RULE model ${HARM_MODEL:-none}, pass-through $PASS, --no-harmonise $((1 - HARMONISE)))"
fi
log "Final reflectance: $RFL"

# ---- Stage 3: writers ---------------------------------------------------------
# GeoTIFF + ENVI BSQ, int16 x10000, water bands dropped, for every sensor (the organisers' submission format), plus
# outlier / snow masks and <name>_provenance.json. --paco-layout also writes EnMAP's PACO-style L2A folder.
stamp write
log "Stage 3/3: writing ${OUT_FORMATS} (int16 x10000, water bands dropped)"
python -m writers --rfl "$RFL" --scene "${WORK}/scene.json" --out "$OUTPUT_DIR" --formats "$OUT_FORMATS" \
    ${REPORT:+--report "$REPORT"} || die "writers failed"
if [[ "$PACO" -eq 1 && "$FMT" == "enmap_l1c" && "$PASS" == "0" ]]; then
    log "Stage 3b: PACO-style L2A layout (--paco-layout)"
    python -m postprocess_to_paco --isofit-dir "${WORK}/isofit" --l1c-dir "$INPUT" --output "$OUTPUT_DIR" \
        --season "$SEASON" --emulator "$EMU" --atm-post "$REPORT" --pipeline-version "${PIPELINE_VERSION:-unknown}"
fi

stamp end
# The HTML quick-look report (report.py) reads the work dir, so it runs before the clean-up. It is optional: a
# failure is logged and the products stand.
if [[ "$REPORT_HTML" -eq 1 ]]; then
    log "Stage 3c: HTML quick-look report"
    python -m report --out "$OUTPUT_DIR" --work "$WORK" "${REF_ARGS[@]}" \
        || log "WARNING: report failed; the products are unaffected"
fi

if [[ "$KEEP_WORK" -eq 0 ]]; then
    log "Removing intermediate work dir (pass --keep-work to retain)"
    rm -rf "$WORK"
fi
log "DONE. L2A products in: $OUTPUT_DIR"
