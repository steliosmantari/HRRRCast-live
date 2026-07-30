#!/usr/bin/env bash
#
# aws/run_subdomain_forecast.sh — run one HRRRCast forecast on a cropped
# subdomain, on this instance.
#
# Sibling of aws/domain_test.sh, but for production use rather than a
# full-vs-sub comparison: it runs the input stages ONCE at full domain (the
# pipeline's regridding is fixed at 1059x1799; see src/crop_domain.py), crops
# to the requested box, and runs a single forecast on the crop. No full-domain
# forecast, no noise yardstick -- just the subdomain you asked for.
#
# Meant to be driven through aws/run_on_ec2.sh --run-cmd, which supplies the
# whole bootstrap (code fetch, conda env, log shipping, self-termination) and
# already exports DATAROOT, S3_OUTPUT, NC_COMPLEVEL, NC_LSD, GFS_MIN_LAG,
# MAKE_BCS_WORKERS and PURGE_LOCAL=YES (see aws/user_data.sh). Those are read
# here as environment overrides so this script behaves the same way whether
# it is invoked through run_on_ec2.sh or by hand on an existing box.
#
# Usage:
#   aws/run_subdomain_forecast.sh INIT_TIME LEAD_HOURS [HEIGHT] [WIDTH] [BBOX] [HALO]
#
# Subdomain size is constrained: height % 8 == 3 and width % 8 == 7 (the
# model's fixed reflect-padding; see src/crop_domain.py). BBOX ("N,W,S,E" in
# degrees) overrides HEIGHT/WIDTH and sizes+places the crop to contain that
# region plus HALO cells on every side. Validate a size/box before spending
# money on an instance:
#   python3 src/crop_domain.py --in-dir . --out-dir . --init-time X \
#       --height H --width W --dry-run
#
# Example, via run_on_ec2.sh:
#   aws/run_on_ec2.sh --bucket my-bucket --init-time 2026-07-29T00 \
#       --run-cmd "./aws/run_subdomain_forecast.sh '2026-07-29T00' '24' '' '' '25,-100,35,-80' '40'"
set -uo pipefail

INIT_TIME="${1:?INIT_TIME (YYYY-MM-DDTHH) required}"
LEAD_HOUR="${2:?LEAD_HOURS required}"
SUB_H="${3:-531}"
SUB_W="${4:-903}"
# Optional: size and place the crop from a region of interest instead of H/W.
# Without it the box is centred on the CONUS grid. Format "N,W,S,E" in degrees.
SUB_BBOX="${5:-}"
SUB_HALO="${6:-40}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATAROOT="${DATAROOT:-$(pwd)}"
MODEL="${MODEL:-${REPO_DIR}/net-diffusion/model.keras}"
STATS="${STATS:-${REPO_DIR}/net-diffusion/normalize-stats.nc}"

# Same env-overridable knobs run_cycle.sh exposes, so this script behaves
# consistently whether launched through run_on_ec2.sh or by hand.
NC_COMPLEVEL="${NC_COMPLEVEL:-1}"
NC_LSD="${NC_LSD:-2}"
# 0 = newest GFS cycle. That is wrong for a schedule that fires close to a cycle
# boundary: GFS takes ~4 h to publish while HRRR lands in ~51 min, so a run close
# behind a fresh HRRR cycle can ask for a GFS cycle that has not finished
# publishing yet and fail in get_bcs. Set GFS_MIN_LAG=4 (or run aws/pick_cycle.sh)
# for unattended/scheduled use; see aws/domain_test.sh for the same tradeoff.
GFS_MIN_LAG="${GFS_MIN_LAG:-0}"
S3_OUTPUT="${S3_OUTPUT:-}"
PURGE_LOCAL="${PURGE_LOCAL:-NO}"
NO_GRIB2="${NO_GRIB2:-YES}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
PMM_ALPHA="${PMM_ALPHA:-0.7}"
NOISE_RHO="${NOISE_RHO:-0.9}"
NO_DIFFUSION="${NO_DIFFUSION:-NO}"

DATE="${INIT_TIME%%T*}"; DATE="${DATE//-/}"
HOUR="${INIT_TIME#*T}"
STAMP="${DATE}_${HOUR}"
CYCLE_DIR="${DATAROOT}/${DATE}/${HOUR}"
CROP_DIR="${DATAROOT}/${DATE}/${HOUR}-sub"
mkdir -p "${DATAROOT}/logs"

log() { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

[ -f "$MODEL" ] || die "Model not found: $MODEL"
[ -f "$STATS" ] || die "Normalization stats not found: $STATS"

if [ -n "$SUB_BBOX" ]; then
    log "Subdomain forecast  init=${INIT_TIME}  lead=${LEAD_HOUR}h  bbox=${SUB_BBOX} halo=${SUB_HALO}"
else
    log "Subdomain forecast  init=${INIT_TIME}  lead=${LEAD_HOUR}h  size=${SUB_H}x${SUB_W}"
fi

# Validate the size up front. The same check runs again inside crop_domain.py,
# but on an EC2 box that check comes after ~5 min of input-stage cost.
if [ -z "$SUB_BBOX" ]; then
    python3 - "$SUB_H" "$SUB_W" <<'PY' || die "invalid subdomain size (see above)"
import sys, os
sys.path.insert(0, "src")
from crop_domain import valid_size
h, w = int(sys.argv[1]), int(sys.argv[2])
ok, msg = valid_size(h, w)
if not ok:
    print(f"  {msg}", file=sys.stderr)
sys.exit(0 if ok else 1)
PY
fi

# --- stage 1: inputs, at full domain ---------------------------------------
# The pipeline's regridding weights are fixed at 1059x1799 (see the module
# docstring in src/crop_domain.py), so the input stages always run at full
# domain; only the forecast itself runs on the crop.
log "Input stages (full domain)"
python3 "${REPO_DIR}/src/get_ics.py" "${INIT_TIME}" --base_dir "${DATAROOT}" \
    2>&1 | tee "${DATAROOT}/logs/get-ics.out" || die "get_ics failed"
python3 "${REPO_DIR}/src/get_bcs.py" "${INIT_TIME}" "${LEAD_HOUR}" --base_dir "${DATAROOT}" \
    --gfs_min_lag_hours "${GFS_MIN_LAG}" \
    2>&1 | tee "${DATAROOT}/logs/get-bcs.out" || die "get_bcs failed"
python3 "${REPO_DIR}/src/make_ics.py" "${STATS}" "${INIT_TIME}" \
    --base_dir "${DATAROOT}" --output_dir "${DATAROOT}" \
    2>&1 | tee "${DATAROOT}/logs/make-ics.out" || die "make_ics failed"
python3 "${REPO_DIR}/src/make_bcs.py" "${STATS}" "${INIT_TIME}" "${LEAD_HOUR}" \
    --base_dir "${DATAROOT}" --output_dir "${DATAROOT}" \
    --hrrr_grid_file "${DATE}/${HOUR}/hrrr_${DATE}_${HOUR}_surface.grib2" \
    2>&1 | tee "${DATAROOT}/logs/make-bcs.out" || die "make_bcs failed"

# --- stage 2: crop -----------------------------------------------------------
if [ -n "$SUB_BBOX" ]; then
    log "Cropping to contain ${SUB_BBOX} with a ${SUB_HALO}-cell halo"
    CROP_ARGS=(--bbox "$SUB_BBOX" --halo "$SUB_HALO")
else
    log "Cropping inputs to ${SUB_H}x${SUB_W} (centred on the grid)"
    CROP_ARGS=(--height "$SUB_H" --width "$SUB_W")
fi
python3 "${REPO_DIR}/src/crop_domain.py" --in-dir "$CYCLE_DIR" --out-dir "$CROP_DIR" \
    --init-time "$INIT_TIME" "${CROP_ARGS[@]}" || die "crop_domain failed"

# fcst.py reads its inputs from <base_dir>/<DATE>/<HH>/, so the cropped run
# needs its own DATAROOT whose cycle directory holds the cropped npz, mirroring
# what aws/domain_test.sh does.
SUB_ROOT="${DATAROOT}/subrun"
mkdir -p "${SUB_ROOT}/${DATE}/${HOUR}"
cp "${CROP_DIR}/hrrr_${STAMP}.npz" "${CROP_DIR}/gfs_${STAMP}.npz" "${SUB_ROOT}/${DATE}/${HOUR}/"

# --- stage 3: forecast on the crop -------------------------------------------
FCST_FLAGS=()
[ "$NO_DIFFUSION" == "YES" ] && FCST_FLAGS+=(--no_diffusion)
[ "$NO_GRIB2"     == "YES" ] && FCST_FLAGS+=(--no_grib2)
[ -n "$NC_LSD" ]             && FCST_FLAGS+=(--nc_least_significant_digit "$NC_LSD")
[ -n "$S3_OUTPUT" ]          && FCST_FLAGS+=(--s3_output "${S3_OUTPUT}")
[ "$PURGE_LOCAL"  == "YES" ] && FCST_FLAGS+=(--purge_local)

if [ -n "$SUB_BBOX" ]; then
    log "Forecast (subdomain: bbox ${SUB_BBOX} + ${SUB_HALO}-cell halo)"
else
    log "Forecast (subdomain: ${SUB_H}x${SUB_W})"
fi
python3 "${REPO_DIR}/src/fcst.py" "$MODEL" "$INIT_TIME" "$LEAD_HOUR" \
    --num_members 1 --members 0 --batch_size "$BATCH_SIZE" --log_level "$LOG_LEVEL" \
    --pmm_alpha "$PMM_ALPHA" --noise_rho "$NOISE_RHO" \
    --nc_complevel "$NC_COMPLEVEL" \
    ${FCST_FLAGS[@]+"${FCST_FLAGS[@]}"} \
    --base_dir "$SUB_ROOT" --output_dir "$SUB_ROOT" \
    2>&1 | tee "${DATAROOT}/logs/fcst.out"
RC=${PIPESTATUS[0]}
[ "$RC" -eq 0 ] || die "fcst failed (rc=${RC}, see ${DATAROOT}/logs/fcst.out)"

cp "${CROP_DIR}/subdomain.json" "${SUB_ROOT}/" 2>/dev/null || true
log "Subdomain forecast complete. Outputs under ${SUB_ROOT}/${DATE}/${HOUR}/  (logs in ${DATAROOT}/logs/)"
